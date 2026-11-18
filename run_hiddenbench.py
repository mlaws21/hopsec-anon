import sys
import argparse
import glob
import json
import re
import logging
import colorlog
import datetime
import os
import random
import time 
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")
os.environ["HF_HUB_CACHE"] = os.path.expanduser("~/hf_cache/hub")

from config.setting import *
from src.experiment import Experiment
from src.manual_backend import load_manual_script, set_agent_abilities
from src.utils import read_yaml


from dotenv import load_dotenv
load_dotenv()

# Agent i holds the hidden facts for knowledge_i.
OWN_KNOWLEDGE = 1.0
OTHER_KNOWLEDGE = 0.5   # keep >= 0.5 or routing checks reject every pick

def ability_dims(num_agents):
    return [f"knowledge_{i}" for i in range(num_agents)]


NORMAL_LLM_KEY = os.getenv("AGENT_MODEL", "")



CONFIDENCE_OVERRIDES = {
    # 0: 1.0,
    # 0: {"mathematical": 0.9, "reasoning": 0.8},
}

def abilities_for_agent(agent_id, overrides, dims):
    base = {d: OTHER_KNOWLEDGE for d in dims}
    base[f"knowledge_{agent_id}"] = OWN_KNOWLEDGE
    override = overrides.get(agent_id)
    if isinstance(override, dict):
        base.update({d: v for d, v in override.items() if d in base})
    elif override is not None:
        base = {d: override for d in dims}
    return base


def apply_confidence_overrides(abilities_by_agent, overrides):
    for agent_id, override in overrides.items():
        abilities = abilities_by_agent.get(agent_id)
        if abilities is None:
            continue
        if isinstance(override, dict):
            abilities.update({d: v for d, v in override.items() if d in abilities})
        else:
            abilities.update({d: override for d in abilities})


def build_agent_config(num_agents, confidence_overrides, normal_llm, agent_defaults, dims):

    agent_config = []
    for agent_id in range(num_agents):
        agent_config.append({
            "id": agent_id,
            "config": {
                "LLM": normal_llm,
                "abilities": abilities_for_agent(agent_id, confidence_overrides, dims),
                **agent_defaults,
            },
        })
    return agent_config


def pick_manual_agents(seed, dataset_index, trial, num_agents, num_manual):

    rng = random.Random(f"{seed}:{dataset_index}:{trial}")
    ids = sorted(rng.sample(range(num_agents), num_manual))
    return {agent_id: f"manual-{slot}" for slot, agent_id in enumerate(ids)}


def setup_logging(log_file_path, level="RESULT", mode="w"):
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'RESULT': 'bold_purple',
            'FORWARD': 'bold_green',
            'EXECUTE': 'bold_red',
            'SPLIT': 'bold_yellow',
            'TASK': 'bold_cyan',
            'WARNING': 'yellow,bg_white',
            'ERROR': 'red,bg_white',
            'CRITICAL': 'bold_red,bg_white'
        },
    ))

    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)


def get_args():
    parser = argparse.ArgumentParser(description="Run AgentNet with under different configurations as well as manual agents.")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Name of experiment (default: <model>test<num_test><script>, see default_experiment_name)")
    parser.add_argument("--experiment_config", type=str, default="hiddenbench_config.yaml",
                         help="YAML file in config/experiment/ providing general configurations.")
    parser.add_argument("--script", type=str,
                         help="json file in attacks/ that outlines how an agents behaves")
    parser.add_argument("--num_agents", type=int, default=3, help="Total number of agents in the run")
    parser.add_argument("--num_manual", type=int, default=1,
                         help="How many of the agents are manually-scripted.")
    parser.add_argument("--num_train", type=int, default=0,
                    help="Tasks used for training")
    parser.add_argument("--num_test", type=int, default=3, help="Number of test tasks to run")
    parser.add_argument("--num_trials", type=int, default=1,
                        help="Run each test task this many times, each with its own randomly chosen manual agent(s)")
    parser.add_argument("--seed", type=int, default=0, help="Seed for choosing the manual agent(s) of each test run")
    parser.add_argument("--conf", action='store_true',
                        help="Give each test run's manual agent(s) confidence 1.0 on every ability")
    parser.add_argument("--parallel", action='store_true', help="Run all test tasks concurrently")
    parser.add_argument("--max_workers", type=int, default=0,
                        help="With --parallel, cap on concurrent test tasks (0 = all at once)")
    parser.add_argument("--train_only", action='store_true',
                        help="Run fit() on --num_train tasks, save training_state.pkl, and skip testing")
    parser.add_argument("--no_vtrace", action='store_true',
                        help="Skip vtrace signing/verification/logging and vet/ dumps during fit() "
                             "(chain state still flows through the VET, unsigned)")
    parser.add_argument("--load_training", type=str, default=None,
                        help="training_state.pkl (or the save dir holding it) from an earlier run; "
                             "skips fit() and tests on the tasks after the ones it trained on")
    parser.add_argument("--resume", nargs="?", const="latest", default=None,
                        help="Continue an interrupted run: the newest log/<experiment_name>_<ts>/ run "
                             "(or the given run dir), skipping finished tasks. Result-affecting args "
                             "come from that run; --parallel/--max_workers may change.")
    parser.add_argument("--weaker", action='store_true', help="additional prompting to help weaker models solve tasks.")
    parser.add_argument("--user", type=str, default="darrow", help="user issuing query")
    parser.add_argument("--timelimit", type=int, default=600, help="time limit for task in seconds")
    
    
    
    args = parser.parse_args()
    return args, parser


RESUME_ARGS = ["experiment_config", "script", "num_agents", "num_manual", "num_train",
               "num_test", "num_trials", "seed", "conf", "train_only", "load_training"]
RUN_CONFIG_FILENAME = "run_config.json"


def default_experiment_name(args):
    model = NORMAL_LLM_KEY.rsplit("/", 1)[-1]
    return f"{model}_test_{args.num_test}x{args.num_trials}_{args.script or ''}"


def find_resume_dir(resume, experiment_name):
    if resume != "latest":
        if not os.path.isfile(os.path.join(resume, RUN_CONFIG_FILENAME)):
            raise FileNotFoundError(f"--resume {resume}: no {RUN_CONFIG_FILENAME} in that directory")
        return resume
    plain = os.path.join("./log", experiment_name)
    if os.path.isfile(os.path.join(plain, RUN_CONFIG_FILENAME)):
        return plain
    pattern = re.compile(rf"^{re.escape(experiment_name)}_\d{{4}}-\d{{6}}$")
    runs = sorted(d for d in glob.glob(os.path.join("./log", f"{glob.escape(experiment_name)}_*"))
                  if pattern.match(os.path.basename(d))
                  and os.path.isfile(os.path.join(d, RUN_CONFIG_FILENAME)))
    if not runs:
        raise FileNotFoundError(f"--resume: no previous run named {experiment_name!r} under ./log")
    return runs[-1]


def merge_resume_args(args, parser, saved):
    for name in RESUME_ARGS:
        # Runs saved before an arg existed used its default.
        cli, default = getattr(args, name), parser.get_default(name)
        prev = saved["args"].get(name, default)
        if cli != default and cli != prev:
            parser.error(f"--{name} {cli!r} conflicts with the resumed run's value {prev!r}")
        setattr(args, name, prev)
    args.experiment_name = saved["experiment_name"]


def script_has_ordered_state(script_dir_path, script):
    if not script:
        return False
    with open(os.path.join(script_dir_path, f"{script}.json")) as f:
        text = f.read()
    return any(isinstance(v, list) for v in json.loads(text).values()) or "round_robin" in text


def main():
    
    args, parser = get_args()
    if args.experiment_name is None:
        args.experiment_name = default_experiment_name(args)

    saved_run = None
    if args.resume:
        resume_dir = find_resume_dir(args.resume, args.experiment_name)
        with open(os.path.join(resume_dir, RUN_CONFIG_FILENAME)) as f:
            saved_run = json.load(f)
        merge_resume_args(args, parser, saved_run)
        date = saved_run["timestamp"]
    else:
        date = datetime.datetime.now().strftime('%m%d-%H%M%S')

    os.makedirs("./log", exist_ok=True)
    log_file_path = os.path.join("./log", f"{args.experiment_name}.log")
    trace_dir = os.path.join("./log", f"{args.experiment_name}")
    os.makedirs(trace_dir, exist_ok=True)
    setup_logging(log_file_path, mode="a" if saved_run else "w")
    logger = logging.getLogger(__name__)
    logger.info("Progress Start!")
    if saved_run:
        logger.result(f"=== Resumed {datetime.datetime.now():%Y-%m-%d %H:%M:%S} ===")
        if saved_run.get("normal_llm") != NORMAL_LLM_KEY:
            logger.warning(f"Resumed run used benign LLM {saved_run.get('normal_llm')!r}, "
                           f"this session uses {NORMAL_LLM_KEY!r}")
    else:
        with open(os.path.join(trace_dir, RUN_CONFIG_FILENAME), "w") as f:
            json.dump({"experiment_name": args.experiment_name, "timestamp": date,
                       "normal_llm": NORMAL_LLM_KEY,
                       "args": {name: getattr(args, name) for name in RESUME_ARGS}}, f, indent=4)

    # setting up additional logging
    json_file_path_result = f"./save/{args.experiment_name}_{date}/result.json"
    json_file_path_ability = f"./save/{args.experiment_name}_{date}/ability.json"
    json_file_path_edge_weight = f"./save/{args.experiment_name}_{date}/edge_weight.json"
    json_file_path_task_history = f"./save/{args.experiment_name}_{date}/task_history.json"
    json_file_path_experience = f"./save/{args.experiment_name}_{date}/experience.json"

    os.makedirs(os.path.dirname(json_file_path_result), exist_ok=True)
    os.makedirs(os.path.dirname(json_file_path_ability), exist_ok=True)
    os.makedirs(os.path.dirname(json_file_path_edge_weight), exist_ok=True)
    os.makedirs(os.path.dirname(json_file_path_task_history), exist_ok=True)
    os.makedirs(os.path.dirname(json_file_path_experience), exist_ok=True)


    config_dir_path = "./config/experiment"
    script_dir_path = "./attacks"

    total_experiment_config = read_yaml(os.path.join(config_dir_path, args.experiment_config))

    experiment_config = total_experiment_config["experiment_config"]
    agent_graph_config = total_experiment_config["agent_graph_config"]
    experiment_config["global_router_experience"] = False
    agent_graph_config["global_router_experience"] = False


    assert 0 <= args.num_manual <= args.num_agents, "--num_manual can't exceed --num_agents"
    experiment_config["agent_num"] = args.num_agents
    
    if experiment_config.get("forward_path_max_length", 0) <= 0:
        experiment_config["forward_path_max_length"] = args.num_agents - 1

    assert args.num_trials >= 1, "--num_trials must be at least 1"

    dims = ability_dims(args.num_agents)
    import config.setting as setting
    setting.task_to_ability_map["hidden_profile"] = dims

    agent_config = build_agent_config(
        num_agents=args.num_agents,
        confidence_overrides=CONFIDENCE_OVERRIDES,
        normal_llm=NORMAL_LLM_KEY,
        agent_defaults=total_experiment_config["agent_defaults"],
        dims=dims,
    )

    set_agent_abilities({agent["id"]: agent["config"]["abilities"] for agent in agent_config})

    if args.num_manual > 0:
        load_manual_script(script_dir_path, args.script,
                           required_slots=[f"manual-{slot}" for slot in range(args.num_manual)])

    from evaluator.datasets.hiddenbench import HiddenBenchDataset
    from src.checkpoint import read_training_state, CHECKPOINT_FILENAME

    training_state = None
    num_train = args.num_train
    if args.load_training:
        assert not args.train_only, "--train_only and --load_training are mutually exclusive"
        assert args.num_train == 0, "--num_train comes from the checkpoint when using --load_training"
        training_state = read_training_state(args.load_training)
        train_meta = training_state["meta"]
        assert train_meta.get("num_train_completed", train_meta["num_train"]) == train_meta["num_train"], \
            (f"{training_state['path']} is a partial checkpoint ({train_meta['num_train_completed']}/"
             f"{train_meta['num_train']} train tasks); finish it with --resume instead of loading it")
        assert train_meta.get("dataset") == "hiddenbench", \
            f"{training_state['path']} was trained on {train_meta.get('dataset')!r}, not hiddenbench"
        assert train_meta["num_agents"] == args.num_agents, \
            f"checkpoint was trained with --num_agents {train_meta['num_agents']}, this run has {args.num_agents}"
        num_train = train_meta["num_train"]
    if args.train_only:
        assert args.num_train > 0, "--train_only needs --num_train > 0"

    dataset = HiddenBenchDataset(num_agents=args.num_agents)
    all_tasks = dataset.generate_tasks()
    train_dataset = [] if training_state else all_tasks[:num_train]
    test_dataset = [] if args.train_only else all_tasks[num_train:num_train + args.num_test]


    for task in train_dataset:
        task.user_id = args.user
        task.timelimit = args.timelimit
        
    print(f"Test Instances: {len(test_dataset)}")
    
    from prompt.hiddenbench_prompt_set import HiddenBenchPromptSet
    prompt_set = HiddenBenchPromptSet()
    constraints = prompt_set.get_constraint()
    thought_constraints = prompt_set.get_thought_constraint()
    split_constraints = prompt_set.get_split_constraint()

    experiment = Experiment(experiment_config,
                            agent_config,
                            agent_graph_config,
                            train_dataset,
                            test_dataset,
                            json_file_path_task_result=json_file_path_result,
                            json_file_path_agent_info=json_file_path_ability,
                            json_file_path_edge_weight=json_file_path_edge_weight,
                            json_file_path_task_history=json_file_path_task_history,
                            json_file_path_experience=json_file_path_experience,
                            constraints=constraints,
                            thought_constraints=thought_constraints,
                            split_constraints=split_constraints,
                            trace_dir=trace_dir)

    import src.agent as agent_mod
    import src.manual_backend as manual_mod
    router_suffix = (
        "\nOther agents hold private information that is necessary to solve this task, "
        "and deciding without it is likely to be wrong. "
        "Before executing, make sure the Progress includes what the other agents know. "
        "Executing ends the task immediately with your answer, and other agents will never see your message. "
        "To share information or hear from others, you must split or forward."
    )
    
    if args.weaker:
        for mod in (setting, agent_mod, manual_mod):
            mod.ROUTER_SYSTEM_PROMPT += router_suffix
    
    if training_state is not None:
        experiment.load_training_state(training_state)
        if CONFIDENCE_OVERRIDES:
            apply_confidence_overrides(
                {agent_id: agent.abilities for agent_id, agent in experiment.agent_graph.agents.items()},
                CONFIDENCE_OVERRIDES)
            logger.result(f"Re-applied confidence overrides on top of loaded abilities: {CONFIDENCE_OVERRIDES}")
        logger.result(f"Loaded training state {training_state['path']} "
                      f"({train_meta['num_train']} train tasks, train accuracy {train_meta.get('train_accuracy')}, "
                      f"trained LLMs {train_meta.get('agent_llms')}); testing on tasks "
                      f"{num_train + 1}-{num_train + len(test_dataset)}")

    checkpoint_path = os.path.join(os.path.dirname(json_file_path_result), CHECKPOINT_FILENAME)

    def save_checkpoint(num_completed):
        """After every train task, so --resume can continue mid-training."""
        experiment.save_training_state(checkpoint_path, meta={
            "experiment_name": args.experiment_name,
            "dataset": "hiddenbench",
            "num_agents": args.num_agents,
            "num_train": len(train_dataset),
            "num_train_completed": num_completed,
            "train_task_ids": [task.task_id for task in train_dataset],
            "agent_llms": {agent["id"]: agent["config"]["LLM"] for agent in agent_config},
            "script": args.script if args.num_manual > 0 else None,
            "manual": "none in training; random per test run",
            "conf": args.conf,
            "train_accuracy": experiment.train_accuracy.get_accuracy(),
        })

    train_done, test_done = 0, set()
    if saved_run:
        if train_dataset and os.path.exists(checkpoint_path):
            partial = read_training_state(checkpoint_path)
            train_done = partial["meta"].get("num_train_completed", partial["meta"]["num_train"])
            experiment.load_training_state(partial)
            experiment.restore_train_progress(train_done)

        from src.run_trace import record_index
        def finished(record):
            if not record.get("completed"):
                return False
            if record["mode"] == "Train":
                return record_index(record) <= train_done
            return record_index(record) <= len(test_dataset)
        kept = experiment.run_trace.restore(finished)
        test_results = [r for r in kept if r["mode"] == "Test"]
        experiment.restore_test_progress(test_results)
        test_done = {r["file"] for r in test_results}

        logger.result(f"Resuming {trace_dir}: train {train_done}/{len(train_dataset)} done, "
                      f"test {len(test_done)}/{len(test_dataset) * args.num_trials} runs done")
        if script_has_ordered_state(script_dir_path, args.script if args.num_manual > 0 else None):
            logger.warning(f"Manual script {args.script!r} uses a sequence or round_robin; "
                           f"its cursor restarts from the beginning on --resume")

    if train_dataset:
        if train_done < len(train_dataset):
            experiment.fit(start_index=train_done, after_task=save_checkpoint,
                           vtrace=not args.no_vtrace)
        logger.result(f"Saved training state to {checkpoint_path}")

    def prepare_trial(task, i, trial):

        task.trial = trial
        task.manual_agents = {}
        if args.num_manual > 0:
            task.manual_agents = pick_manual_agents(
                args.seed, num_train + i, trial, args.num_agents, args.num_manual)
        
        task.user_id = args.user
        task.timelimit = args.timelimit
        
        if args.conf:
            task.ability_overrides = {agent_id: 10.0 for agent_id in task.manual_agents}

    if len(test_done) < len(test_dataset) * args.num_trials:
        experiment.evaluate(parallel=args.parallel, max_workers=args.max_workers or None, skip=test_done,
                            num_trials=args.num_trials, prepare_trial=prepare_trial)
    elif test_dataset:
        logger.result(f"All {len(test_dataset) * args.num_trials} test runs already finished; "
                      f"final accuracy {experiment.test_accuracy.get_accuracy()}")


if __name__ == "__main__":
    main()

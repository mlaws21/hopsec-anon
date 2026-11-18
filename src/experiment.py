import os
import copy
import time
import logging
import threading
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from . import log_utils
from . import run_trace
from . import timing
from . import checkpoint
from . import manual_backend
from .agentgraph import AgentGraph
from .agent import is_manual_agent
from .task import TaskChain
from .utils import save_json, load_json
from prompt.default import DefaultPromptSet
from .utils import extract_answer
from evaluator.Accuracy import Accuracy
from evaluator.evaluator import evaluate_answer
logger = logging.getLogger(__name__)

from .vtrace import USERS, generate_ed25519_keypair, user_name, agent_name, add_step, check_incoming_both


class Experiment:
    def __init__(self, 
                 experiment_config, 
                 agent_config, 
                 agent_graph_config,
                 train_dataset,
                 test_dataset,

                 json_file_path_task_result,
                 json_file_path_agent_info,
                 json_file_path_edge_weight,
                 json_file_path_task_history,
                 json_file_path_experience,

                 constraints = DefaultPromptSet.get_constraint(),
                 split_constraints = DefaultPromptSet.get_split_constraint(),
                 thought_constraints = DefaultPromptSet.get_thought_constraint(),

                 trace_dir=None):
        self.experiment_config = experiment_config
        self.forward_path_max_length = experiment_config["forward_path_max_length"]
        self.max_execution_times = experiment_config["max_execution_times"]
        self.max_hops = experiment_config["max_hops"]
        
        self.use_react = experiment_config["use_react"] 
        self.agent_config = agent_config
        self.agent_graph_config = agent_graph_config

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.constraints = constraints
        self.json_file_path_task_result = json_file_path_task_result
        self.json_file_path_agent_info = json_file_path_agent_info
        self.json_file_path_edge_weight = json_file_path_edge_weight
        self.json_file_path_task_history = json_file_path_task_history
        self.json_file_path_experience = json_file_path_experience


        self.task_result_record = []
        self.agent_ability_record = []
        self.edge_weight_record = []
        self.agent_info_record = []
        self.agent_experiences_record = []
        self.task_history_record = []

        self.train_accuracy = Accuracy()
        self.test_accuracy = Accuracy()
        self.train_agents_visited_total = 0
        self.train_executes_total = 0
        self.test_agents_visited_total = 0
        self.test_executes_total = 0

        self.train_runtime_total = 0.0
        self.train_non_manual_runtime_total = 0.0
        self.train_non_manual_visits_total = 0
        self.train_non_manual_executes_total = 0
        self.test_runtime_total = 0.0
        self.test_non_manual_runtime_total = 0.0
        self.test_non_manual_visits_total = 0
        self.test_non_manual_executes_total = 0

        self.train_timing_total = {}
        self.test_timing_total = {}

        self.thought_constraints = thought_constraints
        self.split_constraints = split_constraints
        self.agent_graph = self.initilize_agent_graph(agent_config, agent_graph_config)
        self.task_history = {}
        self._eval_lock = threading.Lock()

        self.vtrace_enabled = True


        #### PKI ####
        
        self.user_sks = {}
        for u in USERS:
            sk, pk = generate_ed25519_keypair()
            self.user_sks[user_name(u)] = sk
            self.agent_graph.pks[user_name(u)] = pk
        
        # dir to save the VET in
        self.vtrace_dir = os.path.join(trace_dir, "vet") if trace_dir else None
        if self.vtrace_dir:
            os.makedirs(self.vtrace_dir, exist_ok=True)
        
        # saving PKs for auditing after
        from cryptography.hazmat.primitives import serialization
        with open(os.path.join(self.vtrace_dir, "pks.json"), "w") as f:
            json.dump({name: pk.public_bytes(serialization.Encoding.Raw,
                                            serialization.PublicFormat.Raw).hex()
                    for name, pk in self.agent_graph.pks.items()}, f, indent=2)
        #### PKI ####

        # Opt-in per-task trace files + token accounting
        self.run_trace = None
        if trace_dir is not None:
            self.run_trace = run_trace.RunTrace(
                trace_dir, {agent["id"]: agent["config"]["LLM"] for agent in agent_config})
            run_trace.activate(self.run_trace)
                

    def initilize_agent_graph(self, agent_config, agent_graph_config):
        return AgentGraph(agent_config, agent_graph_config)


    @staticmethod
    def _task_label(task_chain):
        """"Task <id>", plus the trial when a task is run several times."""
        trial = getattr(task_chain.original_task, "trial", None)
        return f"Task {task_chain.task_chain_id}" + (f" (trial {trial})" if trial is not None else "")

    @staticmethod
    def _is_manual_agent(agent):
        return is_manual_agent(agent.agent_id, agent.llm)

    def _resolve_next_executor(self, task_chain, current_agent, next_agent_id):

        if next_agent_id is not None and next_agent_id in self.agent_graph.agents:
            return self.agent_graph.agents[next_agent_id]
        fallback_id = self.agent_graph.select_an_agent(task_chain.task_type)
        return self.agent_graph.agents[fallback_id]

    def _force_final_execute(self, task_chain, current_agent, reason):

        task_chain.set_current_task_remaining_work(
            "All required subtasks are believed complete. Synthesize and "
            "provide the final answer to the Major Task using the Completed "
            f"Subtasks and Results so far. {reason}"
        )
        output = current_agent.execute_task(
            task_chain, constraints=self.constraints, split_constraints=self.split_constraints,
            thought_constraints=self.thought_constraints, use_react=self.use_react, decision="execute")

        logger.debug(f"Forced Final Executor Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
        logger.debug(f"Forced Final Executor Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")

        task_chain.add_task_history(
            format_response=output["response_dict"],
            original_response=output["response"],
            agent_id=current_agent.agent_id,
            experience=output["experience"],
            execution_time=output["execution_time"],
            mode="Execution")

        task_chain.set_current_task_result(
            agent_id=current_agent.agent_id,
            result=output["result"])

        self._add_step(task_chain, agent_name(current_agent.agent_id), task_chain.user_id, None, current_agent.sk, current_agent.most_recent_trace)

    def _add_step(self, task_chain, sender, receiver, remaining_work, sk, most_recent_trace=None):
        """vtrace.add_step, unsigned and unlogged when vtrace is off."""
        if not self.vtrace_enabled:
            sk, most_recent_trace = None, None
        return add_step(task_chain, sender, receiver, remaining_work, sk, most_recent_trace)

    def _hand_off(self, task_chain, sender, receiver_name, remaining_work):

        self._add_step(task_chain, agent_name(sender.agent_id), receiver_name, remaining_work, sender.sk, sender.most_recent_trace)
        task_chain.receive()

    def _check_incoming(self, task_chain, agent):

        if not self.vtrace_enabled:
            return
        verdicts = check_incoming_both(task_chain.vet, agent_name(agent.agent_id), self.agent_graph.pks)
        prev = task_chain.vet[-1][0]["sender"]
        for mode, (ok, why) in verdicts.items():
            if ok:
                continue
            logger.warning(f"vtrace reject (verify_{mode}) at {agent.agent_id}: {why}")
            attr = f"was_tampered_{mode}"
            if getattr(task_chain, attr) is None:
                setattr(task_chain, attr, {"blame": (prev, agent_name(agent.agent_id)), "reason": why})

    def solve_a_single_task(self, task):
        """main driver of all tasks"""
        
        # chain keeping track of the solving of a task
        task_chain = TaskChain(task)

        # none means set allowed agents to all
        if task_chain.allowed_agents is None:
            task_chain.allowed_agents = [agent_name(a) for a in list(self.agent_graph.agents)]
        # progress entries read from the VET carry the raw agent id again
        task_chain.agent_id_of = {agent_name(a): a for a in self.agent_graph.agents}

        # set of manual agents
        manual_agents = getattr(task, "manual_agents", None)
        
        # sets higher conf
        manual_backend.set_task_context(
            task_chain, manual_agents, getattr(task, "ability_overrides", None))
        
        # adversaries know the correct answer so they can avoid (if they choose to)
        manual_backend.set_correct_answer(task.correct_answer)
        
        # sets some max path numbers
        forward_times = self.forward_path_max_length
        current_execution_times = 0
        current_hops = 0
        
        # sets first agent
        current_agent_id = self.agent_graph.select_an_agent(task.task_type)
        current_agent = self.agent_graph.agents[current_agent_id]
        
        task_chain.user_id = user_name(task_chain.user_id)
        
        self._add_step(task_chain, task_chain.user_id, agent_name(current_agent_id), task_chain.major_problem, self.user_sks[task_chain.user_id])
        task_chain.receive()

        # timing and visits logging
        task_start_time = time.time()
        timing.start()
        manual_time = 0.0
        non_manual_visits = 0
        non_manual_executes = 0

        # task label for the logs
        task_label = self._task_label(task_chain)
        if manual_agents:
            task_label += f" (manual agents {sorted(manual_agents)})"
        logger.task("="*8 + f"{task_label} Starts" + "="*8)

        while True:
            turn_start_time = time.time()
            deciding_agent = current_agent
            is_manual_turn = self._is_manual_agent(deciding_agent)
            executed_this_turn = False
            try:
                # Lets a manual "revert_at_hops" step back out at its hop count.
                manual_backend.on_hop(current_hops)
                logger.info(f"Agent {current_agent.agent_id} is Solving Task {task_chain.get_current_task_id()}")
                
                self._check_incoming(task_chain, current_agent)

                output = current_agent.decide_action(task_chain, forward_times)
                logger.debug(f"Router Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
                logger.debug(f"Router Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")
                logger.route(current_agent.agent_id, output["decision"], output["response_dict"].get("REASON", ""))

                task_chain.add_task_history(
                    format_response=output["response_dict"],
                    original_response=output["response"],
                    agent_id=current_agent.agent_id,
                    experience=output["experience"],
                    execution_time=output["execution_time"],
                    mode="Router_Decision")

                decision = output["decision"]
                next_agent_id = output["next_agent_id"]
                executable_tasks = output["executable_tasks"]
                description = output["description"]

                current_hops += 1
                
                if current_hops >= self.max_hops:
                    logger.route(
                        current_agent.agent_id, "execute",
                        "At max_hops; executing.")
                    self._force_final_execute(
                        task_chain, current_agent,
                        reason="The maximum number hops was reached, so this agent executed.")
                    executed_this_turn = True
                    break
                
                if decision in ["forward"]:
                    logger.info(f"Task {task_chain.get_current_task_id()} is Forwarded from Agent {current_agent.agent_id} to Agent {next_agent_id}")

                    task_chain.current_task.add_forward_history(current_agent.agent_id)
                    
                    self._hand_off(task_chain, current_agent, agent_name(next_agent_id),
                                   task_chain.current_task.remaining_work)

                    current_agent = self.agent_graph.agents[next_agent_id]
                    forward_times -= 1

                    task_chain.add_task_history(
                        format_response=output["response_dict"],
                        original_response=output["response"],
                        experience=output["experience"],
                        agent_id=current_agent.agent_id,
                        execution_time=output["execution_time"],
                        mode="Forward")


                elif decision == "execute":

                    task_chain.set_current_task_remaining_work(description)
                    output = current_agent.execute_task(
                        task_chain, constraints=self.constraints, split_constraints=self.split_constraints,
                        thought_constraints=self.thought_constraints, use_react=self.use_react, decision="execute")

                    logger.debug(f"Executor Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
                    logger.debug(f"Executor Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")

                    task_chain.add_task_history(
                        format_response=output["response_dict"],
                        original_response=output["response"],
                        agent_id=current_agent.agent_id,
                        experience=output["experience"],
                        execution_time=output["execution_time"],
                        mode="Execution")

                    task_chain.set_current_task_result(
                        agent_id=current_agent.agent_id,
                        result=output["result"])

                    self._add_step(task_chain, agent_name(current_agent.agent_id), task_chain.user_id, None, current_agent.sk, current_agent.most_recent_trace)

                    executed_this_turn = True
                    break


                elif decision in ["split"]:

                    # At the cap we no longer allow another split: log an
                    # EXECUTE line noting the limit and execute here instead.
                    if current_execution_times >= self.max_execution_times:
                        logger.route(
                            current_agent.agent_id, "execute",
                            "At max_execution_times; executing instead of splitting.")
                        self._force_final_execute(
                            task_chain, current_agent,
                            reason="The maximum number of split/execution cycles was reached, so this agent executed instead of splitting.")
                        executed_this_turn = True
                        break

                    task_chain.set_current_task_remaining_work(executable_tasks)
                    output = current_agent.execute_task(
                        task_chain, constraints=self.constraints, split_constraints=self.split_constraints,
                        thought_constraints=self.thought_constraints, use_react=self.use_react, decision="split")

                    logger.debug(f"Split Executor Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
                    logger.debug(f"Split Executor Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")

                    task_chain.add_task_history(
                        format_response=output["response_dict"],
                        original_response=output["response"],
                        agent_id=current_agent.agent_id,
                        experience=output["experience"],
                        execution_time=output["execution_time"],
                        mode="Split_Execution")

                    task_chain.set_current_task_result(
                        agent_id=current_agent.agent_id,
                        result=output["result"])

                    executed_this_turn = True

                    task_chain.create_next_task()

                    output = current_agent.decide_next_agent_id(task_chain,self.constraints)

                    logger.debug(f"Next Agent Decidision Router Query Prompt of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['query_prompt']}")
                    logger.debug(f"Next Agent Decidision Router Response of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['response']}")
                    logger.info(f"Next Agent Decidision Router Decision of Agent {current_agent.agent_id} for Task {task_chain.get_current_task_id()} is {output['decision']}")

                    task_chain.add_task_history(
                        format_response=output["response_dict"],
                        original_response=output["response"],
                        agent_id=current_agent.agent_id,
                        experience=output["experience"],
                        execution_time=output["execution_time"],
                        mode="Decide_Next_Agent_ID")

                    if output["decision"].lower().strip() in ["completed"]:
                        # We never end on the split itself. The splitting agent
                        # believes the work is done and we are under the cap, so
                        # hand off to the next agent and tell it to execute the
                        # final answer rather than force-executing here.
                        next_agent = self._resolve_next_executor(
                            task_chain, current_agent, output["next_agent_id"])
                        
                        self._hand_off(task_chain, current_agent, agent_name(next_agent.agent_id),
                                       output["reason"])

                        current_agent = next_agent
                        forward_times = self.forward_path_max_length
                        # Skips the loop top, so check + log the handoff here.
                        self._check_incoming(task_chain, current_agent)
                        logger.route(
                            current_agent.agent_id, "execute",
                            "Prior split judged the task complete; this agent executes the final answer.")
                        self._force_final_execute(
                            task_chain, current_agent,
                            reason="A previous agent judged the task complete after its split, so this agent produced the final answer by executing.")
                        break

                    elif output["decision"].lower().strip() in ["incompleted"]:

                        next_agent_id = output["next_agent_id"]
                        
                        self._hand_off(task_chain, current_agent, agent_name(next_agent_id),
                                       output["reason"])

                        current_agent = self.agent_graph.agents[next_agent_id]
                        forward_times = self.forward_path_max_length
                        current_execution_times += 1

                    else:
                        self._hand_off(task_chain, current_agent, agent_name(current_agent.agent_id),
                                       output["reason"])

                else:
                    logger.route(
                        current_agent.agent_id, "execute",
                        f"Unrecognized router decision {decision!r}; executing to avoid a routing loop.")
                    self._force_final_execute(
                        task_chain, current_agent,
                        reason="The router returned no valid decision, so this agent executed to produce a final answer.")
                    executed_this_turn = True
                    break
            finally:
                turn_elapsed = time.time() - turn_start_time
                if is_manual_turn:
                    manual_time += turn_elapsed
                else:
                    non_manual_visits += 1
                    if executed_this_turn:
                        non_manual_executes += 1

        task_chain.runtime = time.time() - task_start_time
        task_chain.timing = timing.finalize(timing.stop(), task_chain.runtime)
        task_chain.non_manual_runtime = task_chain.runtime - manual_time
        task_chain.non_manual_visits = non_manual_visits
        task_chain.non_manual_executes = non_manual_executes

        return task_chain
    



    def evaluate_task_result(self, task_chain, correct_answer, mode="Train"):
        raw_result = task_chain.get_final_result()
        final_result = extract_answer(raw_result)
        logger.result(f"AgentNet: {final_result.lower().strip()}")
        logger.result(f"Ground Truth: {correct_answer.lower().strip()}")

        success = evaluate_answer(final_result.lower().strip(), correct_answer.lower().strip())
        logger.result(f"{self._task_label(task_chain)} success: {success}")

        agent_path = []
        executed_flags = []
        for h in task_chain.task_history:
            if not agent_path or agent_path[-1] != h.current_agent_id:
                agent_path.append(h.current_agent_id)
                executed_flags.append(False)
            if h.mode in ("Execution", "Split_Execution"):
                executed_flags[-1] = True
        agents_visited = len(agent_path)
        executes_count = sum(executed_flags)

        marked_agent_path = [f"{agent_id}*" if executed else str(agent_id)
                              for agent_id, executed in zip(agent_path, executed_flags)]

        logger.result(f"{self._task_label(task_chain)} path: "
                      f"{marked_agent_path} (* = executed/split)")

        
        runtime = task_chain.runtime
        non_manual_runtime = task_chain.non_manual_runtime
        non_manual_visits = task_chain.non_manual_visits
        non_manual_executes = task_chain.non_manual_executes

        logger.result(f"{self._task_label(task_chain)} [visits] {agents_visited} ({non_manual_visits}), [executes] {executes_count} ({non_manual_executes}), [runtime]: {runtime:.3f}s ({non_manual_runtime:.3f}s)")

        result = {
            "task_chain_id": task_chain.task_chain_id,
            "raw_result": raw_result,
            "final_result": final_result,
            "correct_answer": correct_answer,
            "success": success,
            "agents_visited": agents_visited,
            "agent_path": marked_agent_path,
            "runtime": runtime,
            "non_manual_runtime": non_manual_runtime,
            "non_manual_visits": non_manual_visits,
            "non_manual_executes": non_manual_executes,
            "executes": executes_count,
            "timing": task_chain.timing,
            "rejected": {mode: getattr(task_chain, f"was_tampered_{mode}") is not None for mode in timing.MODES},
        }
        for mode in timing.MODES:
            logger.result(f"{self._task_label(task_chain)} [time {mode}] {timing.format(task_chain.timing[mode])}")

        overall_tokens_str = ""
        if self.run_trace is not None:
            task_tokens = self.run_trace.task_tokens()
            result.update({
                "input_tokens": task_tokens["all"]["input"],
                "output_tokens": task_tokens["all"]["output"],
                "benign_input_tokens": task_tokens["benign"]["input"],
                "benign_output_tokens": task_tokens["benign"]["output"],
                "tokens_by_agent": task_tokens["by_agent"],
            })
            logger.result(f"{self._task_label(task_chain)} [tokens] {run_trace.format_tokens(task_tokens)}")
            overall_tokens_str = f"; [tokens] {run_trace.format_tokens(self.run_trace.cumulative_tokens(mode))}"

        if mode == "Train":
            self.train_accuracy.update(success)
            self.train_agents_visited_total += agents_visited
            self.train_executes_total += executes_count
            self.train_runtime_total += runtime
            self.train_non_manual_runtime_total += non_manual_runtime
            self.train_non_manual_visits_total += non_manual_visits
            self.train_non_manual_executes_total += non_manual_executes
            timing.add_into(self.train_timing_total, task_chain.timing)
            for tmode in timing.MODES:
                logger.result(f"Overall: [time {tmode}] {timing.format(self.train_timing_total.get(tmode))}")
            logger.result(f"Overall: [visits] {self.train_agents_visited_total} ({self.train_non_manual_visits_total}), [executes] {self.train_executes_total} ({self.train_non_manual_executes_total}), [runtime]: {self.train_runtime_total:.1f} ({self.train_non_manual_runtime_total:.1f}){overall_tokens_str}")
            

        if mode == "Test":
            self.test_accuracy.update(success)
            self.test_agents_visited_total += agents_visited
            self.test_executes_total += executes_count
            self.test_runtime_total += runtime
            self.test_non_manual_runtime_total += non_manual_runtime
            self.test_non_manual_visits_total += non_manual_visits
            self.test_non_manual_executes_total += non_manual_executes
            timing.add_into(self.test_timing_total, task_chain.timing)
            for tmode in timing.MODES:
                logger.result(f"Overall: [time {tmode}] {timing.format(self.test_timing_total.get(tmode))}")
            logger.result(f"Overall: [accuracy] {self.test_accuracy.get_accuracy()}; [visits] {self.test_agents_visited_total} ({self.test_non_manual_visits_total}); [executes] {self.test_executes_total} ({self.test_non_manual_executes_total}); [runtime]: {self.test_runtime_total:.1f} ({self.test_non_manual_runtime_total:.1f}){overall_tokens_str}")

        return result


    def update_agent_graph(self, task_chain, result):
        logger.debug("update_agent_graph starts")

        success = result["success"]

        task_history = task_chain.task_history
        for single_task_history in task_history:
            mode = single_task_history.mode
            logger.debug(f"single_task_history.mode is {mode}")
            agent_id = single_task_history.current_agent_id
            agent = self.agent_graph.agents[agent_id]

            experience = single_task_history.experience
            experience.success = success
            # Add current experience to pool
            if mode in ["Execution", "Split_Execution"]:
                agent.add_executor_experience(experience)

            elif mode in ["Decide_Next_Agent_ID", "Router_Decision"]:
                agent.add_router_experience(experience)
                

                # Update edge weights
                source_agent_id = single_task_history.current_agent_id
                logger.debug(f"source_agent_id is {source_agent_id}")
                if "NEXT_AGENT_ID" in single_task_history.format_response.keys():
                    target_agent_id = single_task_history.format_response["NEXT_AGENT_ID"].strip()

                    execution_time = single_task_history.execution_time
                    logger.debug(f"target_agent_id is digit: {target_agent_id.isdigit()}, target_agent_id is {target_agent_id}")
                    if target_agent_id.isdigit():
                        target_agent_id = int(target_agent_id)
                    logger.debug(f"target_agent_id in self.agent_graph.agent_neighbor_dict[source_agent_id]['outcoming_agent_id']: {target_agent_id in self.agent_graph.agent_neighbor_dict[source_agent_id]['outcoming_agent_id']}")
                    if target_agent_id in self.agent_graph.agent_neighbor_dict[source_agent_id]["outcoming_agent_id"]:
                        self.agent_graph.update_edge_weight(source_agent_id, target_agent_id, execution_time, success)
                        logger.info(f"Update Edge Weight from Agent {source_agent_id} to Agent {target_agent_id} to {self.agent_graph.edge_weight[source_agent_id][target_agent_id]}")
                

        
        if success == False:
            logger.info(f"Task {task_chain.task_chain_id} Is Solved Incorrectly, So The Ability of All Agents Will Not Be Updated")
            pass
        elif success == True:
            logger.info(f"Task {task_chain.task_chain_id} Is Solved Correctly, So The Ability of All Agents Will Be Updated")

        task_history_save = []

        for single_task_history in task_history:
            mode = single_task_history.mode
            agent_id = single_task_history.current_agent_id
            agent = self.agent_graph.agents[agent_id]

            if mode in ["Execution", "Split_Execution"]:
                # Update the executing agent's abilities: success strengthens them, failure doesn't lower them
                agent.update_abilities(
                    single_task_history.task.task_type,
                    single_task_history.execution_time,
                    success)

            single_task_history_save = {
                "mode": copy.deepcopy(single_task_history.mode),
                "current_agent_id": copy.deepcopy(single_task_history.current_agent_id),
                "format_response": copy.deepcopy(single_task_history.format_response),
                "original_response": copy.deepcopy(single_task_history.original_response),
                "execution_time": copy.deepcopy(single_task_history.execution_time)
            }
            task_history_save.append(single_task_history_save)


        self.task_history_record.append(task_history_save)                
        result["total_accuracy"] = self.train_accuracy.get_accuracy()
        self.task_result_record.append(result)

        self.edge_weight_record.append({
            "task_chain_id": task_chain.task_chain_id, 
            "edge_weight": self.agent_graph.edge_weight
        })

        all_agent_info = self.agent_graph.get_all_agent_info()
        all_agent_experiences= self.agent_graph.get_all_agent_experiences()

        all_agent_info = copy.deepcopy(all_agent_info)
        self.agent_info_record.append(all_agent_info)

        all_agent_experiences = copy.deepcopy(all_agent_experiences)
        self.agent_experiences_record.append(all_agent_experiences)

        logger.debug(f"The Edge Weight of All Agents Is {self.agent_graph.edge_weight}")

        save_json(self.edge_weight_record, self.json_file_path_edge_weight)
        save_json(self.task_result_record, self.json_file_path_task_result)
        save_json(self.agent_info_record, self.json_file_path_agent_info)
        save_json(self.task_history_record, self.json_file_path_task_history)
        save_json(self.agent_experiences_record, self.json_file_path_experience)


        return 0



    def _solve_and_evaluate(self, task, mode, trace_filename):
        """solve_a_single_task + evaluate_task_result, wrapped in a per-task
        trace file when tracing is on. Scoring holds _eval_lock because it
        updates the shared accuracy/runtime totals (parallel evaluate())."""
        if self.run_trace is not None:
            self.run_trace.start_task(trace_filename, task, mode)
        result = None
        try:
            final_task_chain = self.solve_a_single_task(task)
            with self._eval_lock:
                result = self.evaluate_task_result(final_task_chain, task.correct_answer, mode)
            if hasattr(task, "trial"):
                result["trial"] = task.trial
            if hasattr(task, "manual_agents"):
                result["manual_agents"] = {str(k): v for k, v in task.manual_agents.items()}
            
            if self.vtrace_dir and self.vtrace_enabled:
                path = os.path.join(self.vtrace_dir, trace_filename.replace(".txt", ".json"))
                with open(path, "w") as f:
                    json.dump([{"m": m, "sig": s} for m, s in final_task_chain.vet], f, indent=2)
                # each agent's own sent/recv record for this run (attribution)
                qid = final_task_chain.qid
                with open(path.replace(".json", "_all_traces.json"), "w") as f:
                    json.dump({agent_name(a): agent.most_recent_trace[qid]
                               for a, agent in self.agent_graph.agents.items() if qid in agent.most_recent_trace},
                              f, indent=2)
                for mode in timing.MODES:
                    tampered = getattr(final_task_chain, f"was_tampered_{mode}")
                    if tampered is not None:
                        # which link that verify mode rejected first
                        with open(path.replace(".json", f"_tampered_{mode}.json"), "w") as f:
                            json.dump(tampered, f)
                
        finally:
            timing.stop()  # no-op normally; clears a crashed task's buckets
            manual_backend.clear_task_context()
            manual_backend.set_correct_answer(None)
            if self.run_trace is not None:
                self.run_trace.end_task(result)
        return final_task_chain, result

    def save_training_state(self, path, meta=None):
        """Snapshot everything fit() learned (see src/checkpoint.py)."""
        return checkpoint.save_training_state(self, path, meta)

    def load_training_state(self, path_or_checkpoint):
        """Restore a snapshot saved by save_training_state, in place of fit().
        Accepts a path (file or save dir) or an already-read checkpoint."""
        if isinstance(path_or_checkpoint, dict):
            state = path_or_checkpoint
        else:
            state = checkpoint.read_training_state(path_or_checkpoint)
        return checkpoint.apply_training_state(self, state)

    def _restore_totals(self, mode, results):
        """Re-add already-scored results (from an interrupted run) to the
        accuracy / visit / execute / runtime totals."""
        prefix = mode.lower()
        accuracy = self.train_accuracy if mode == "Train" else self.test_accuracy
        for r in results:

            executes = r.get("executes", sum(str(a).endswith("*") for a in r["agent_path"]))
            accuracy.update(r["success"])
            for name, value in (("agents_visited", r["agents_visited"]), ("executes", executes),
                                ("runtime", r["runtime"]), ("non_manual_runtime", r["non_manual_runtime"]),
                                ("non_manual_visits", r["non_manual_visits"]),
                                ("non_manual_executes", r["non_manual_executes"])):
                attr = f"{prefix}_{name}_total"
                setattr(self, attr, getattr(self, attr) + value)

            timing.add_into(getattr(self, f"{prefix}_timing_total"), r.get("timing"))

    def restore_train_progress(self, num_completed):
        """Resume support: reload the save-JSON records of the first
        num_completed training tasks (truncating any written after the last
        checkpoint) and the train totals they imply."""
        records = [
            ("task_result_record", self.json_file_path_task_result),
            ("agent_info_record", self.json_file_path_agent_info),
            ("edge_weight_record", self.json_file_path_edge_weight),
            ("task_history_record", self.json_file_path_task_history),
            ("agent_experiences_record", self.json_file_path_experience),
        ]
        for attr, path in records:
            saved = load_json(path) if os.path.exists(path) else []
            setattr(self, attr, saved[:num_completed])
        if len(self.task_result_record) != num_completed:
            raise ValueError(f"{self.json_file_path_task_result} has {len(self.task_result_record)} "
                             f"train results, checkpoint says {num_completed} were completed")
        self._restore_totals("Train", self.task_result_record)

    def restore_test_progress(self, results):
        """Resume support: re-add the results of already-finished test tasks."""
        self._restore_totals("Test", results)

    def fit(self, start_index=0, after_task=None, vtrace=True):

        logger.info("Experiment Start in Train Dataset!")
        prev, self.vtrace_enabled = self.vtrace_enabled, vtrace
        try:
            for i, task in enumerate(self.train_dataset, start=1):
                if i <= start_index:
                    continue
                final_task_chain, result = self._solve_and_evaluate(task, "Train", f"train_task{i}.txt")
                self.update_agent_graph(final_task_chain, result)
                if after_task is not None:
                    after_task(i)
        finally:
            self.vtrace_enabled = prev

    def evaluate(self, parallel=False, max_workers=None, skip=(), num_trials=1, prepare_trial=None):

        logger.info("Experiment Start in Test Dataset with the same task type as train dataset !")
        start_time = time.time()
        pending = []
        for i, task in enumerate(self.test_dataset, start=1):
            for t in range(1, num_trials + 1):
                filename = f"task{i}.txt" if num_trials == 1 else f"task{i}_trial{t}.txt"
                if filename in skip:
                    continue
                # A fresh copy per run: solving mutates the task (description,
                # update_task), and every trial must start from the original.
                run_task = copy.deepcopy(task)
                if prepare_trial is not None:
                    prepare_trial(run_task, i, t)
                pending.append((filename, run_task))
        total_runs = len(self.test_dataset) * num_trials

        if not parallel:
            for filename, task in pending:
                final_task_chain, result = self._solve_and_evaluate(task, "Test", filename)
            workers, failures = 1, 0
        else:
            workers = max(1, min(max_workers or len(pending), len(pending)))
            failures = 0
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="task") as pool:
                futures = {pool.submit(self._solve_and_evaluate, task, "Test", filename): filename
                           for filename, task in pending}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception:
                        failures += 1
                        logger.exception(f"Test run {futures[future]} failed")

        wall_clock = time.time() - start_time
        logger.result(f"Test wall-clock: {wall_clock:.1f}s ({len(pending)} runs, {workers} workers"
                      + (f", {total_runs - len(pending)} resumed as done" if skip else "")
                      + (f", {failures} failed" if failures else "") + ")")
        if self.run_trace is not None:
            self.run_trace.set_wall_clock("Test", wall_clock)


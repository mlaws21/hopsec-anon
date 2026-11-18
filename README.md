# Anonymous Repository for Attacks and Defenses in Decentralized Multi-Agent Systems with Hop-by-Hop Task Decomposition and Delegation

## Setup
- setup ure conda env using conda.yml (sorry venv users)
- set up your ```.env``` file we provide an example in ```EXAMPLE.env```
- if you are running a local model you need to host it :)
- note the embedding model uses GPU so it goes faster if you have one

## Training
to train a configuration run this: ```python run_hiddenbench.py --script train --num_agents 4 --num_manual 0 --num_train 20 --num_test 0```. 

you can't run in parallel because they are dependent on previous tasks. 

## Testing

benign: ```python run_hiddenbench.py --script none --num_agents 4 --num_manual 0 --num_train 0 --load_training [point at training pkl] --num_test 20 --parallel --num_trials 5```

malicious: ```python run_hiddenbench.py --script [insert attack] --num_agents 4 --num_manual 1 --num_train 0 --load_training [point at training pkl] --num_test 20 --parallel --num_trials 5```

there are other knobs see (`run_hiddenbench.py`)
## Evaluation

display the results nicely: ```python present_[i|a|c] --model [whatever model u are evaluating]```

for c this doesn't extract the leakage that is done as follows:
- c1: `bash leakage_eval/run_pipeline.sh [log] [training save] c1_measure -k 3 --ignore-case --fold-typography`
- c2: `bash leakage_eval/run_pipeline.sh [log] [training save] c2_measure -k 3 --ignore-case --fold-typography --multihop`
- c3: `python leakage_eval/score_model_evidence.py [log] --truth "[model family] / [model name]"`
- c4 `python leakage_eval/score_prompt_evidence.py [log] --fold-typography`

## Defense

run the auditor like this: `python auditor.py --experiment [log] --all --llm_eval expert` (sender verification)

run the auditor like this: `python auditor.py --experiment [log] --all --verified_all --llm_eval expert` (full-trace verification)

there are other knobs see (`auditor.py`)

## Acknowledgement
This repo is forked from: https://github.com/zoe-yyx/AgentNet
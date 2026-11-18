import random
import threading
import time
import numpy as np
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Union
import copy 
import json
import functools

logger = logging.getLogger(__name__)


from .pool import RouterExperiencePool, ExecutorExperiencePool
from .pool import RouterExperience, ExecutorExperience
from . import run_trace
from . import timing
from .utils import get_gpt_response, get_qwen_response
from .utils import parse_decision_text, extract_agent_id, parse_split_pattern_to_subtask, parse_string_to_dict, extract_content_as_dict
from .task import render_completed_work
from .utils import get_average_abilities_from_task_type


from config.setting import ROUTER_SYSTEM_PROMPT, ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT, EXECUTOR_SYSTEM_PROMPT, THOUGHT_SYSTEM_PROMPT
from config.setting import ROUTER_PROMPT_FORMAT, ROUTER_PROMPT_SPLIT_AND_EXECUTE_FORMAT, ROUTER_PROMPT_DECIDE_NEXT_AGENT_ID_FORMAT
from config.setting import EXECUTOR_PROMPT_FORMAT, THOUGHT_PROMPT_FORMAT, EXECUTOR_THOUGHT_PROMPT_FORMAT, SPLIT_EXECUTOR_PROMPT_FORMAT, SPLIT_THOUGHT_PROMPT_FORMAT
from config.setting import BASIC_PROBLEM_COMPLEXITY, task_to_ability_map

from .vtrace import generate_ed25519_keypair


llm_model_response_map = {
    "gpt-6.1-sol": get_gpt_response,
    "Qwen/Qwen3.8-27B-FP8": get_qwen_response, 
}

from .manual_backend import MANUAL_RESPONSE_FNS
from . import manual_backend

llm_model_response_map.update(MANUAL_RESPONSE_FNS)


def _is_manual_llm(llm):
    return isinstance(llm, str) and llm.startswith("manual-")


def is_manual_agent(agent_id, llm):
    return _is_manual_llm(llm) or manual_backend.manual_key_for(agent_id) is not None


def is_scripted_agent(agent_id, llm):
    return is_manual_agent(agent_id, llm) and not manual_backend.acting_normally(agent_id)

def timed_llm(fn, agent_id, llm):
    @functools.wraps(fn)
    def _wrapped(system_prompt, query_prompt, **kwargs):
        if is_manual_agent(agent_id, llm):
            bucket = "llm_manual"
        else:
            stage = run_trace.stage_for(system_prompt)
            bucket = "llm_other" if stage == "llm" else f"llm_{stage}"
        with timing.timed(bucket):
            return fn(system_prompt=system_prompt, query_prompt=query_prompt, **kwargs)
    return _wrapped


def with_private_info(task, agent_id):
    facts = getattr(task, "private_info", None)
    if not facts or agent_id >= len(facts) or not facts[agent_id]:
        return task.major_problem
    lines = "\n".join(f"- {f}" for f in facts[agent_id])
    return f"{task.major_problem}\n\nPrivate information (only you know this):\n{lines}"

class RouterModule:
    def __init__(self, agent_id, memory_limit, llm, global_router_experience_flag, retrieval_num):
        self.agent_id = agent_id
        self.llm = llm
        self.retrieval_num = retrieval_num
        from .global_vars import global_router_experience_pool

        if global_router_experience_flag == False:
            self.router_pool = RouterExperiencePool(capacity=memory_limit)
        else:
            self.router_pool = global_router_experience_pool

        self.get_llm_response = timed_llm(run_trace.traced(
            manual_backend.task_override(agent_id, llm_model_response_map[llm]), agent_id), agent_id, llm)
        self.get_self_agent_info = None


    def set_get_self_agent_info(self, get_self_info):
        self.get_self_agent_info = get_self_info
        
        

    def router_decide_action(self, task, self_info, neighbors_info, task_history, decide_mode="Normal"):
        start_time = time.time()

        experiences = [] if is_scripted_agent(self.agent_id, self.llm) else self.router_pool.get_relevant_experiences(task, self.agent_id, self.retrieval_num)

        query_prompt = self.router_make_prompt(
            experiences=experiences,
            task=task, 
            decide_mode=decide_mode, 
            self_info=self_info, 
            neighbors_info=neighbors_info
            )
        
        response = self.get_llm_response(
            system_prompt=ROUTER_SYSTEM_PROMPT, 
            query_prompt=query_prompt
            )
        
        response_dict = extract_content_as_dict(long_string=response, short_strings=["DECISION", "REASON", "EXECUTABLE", "DELEGATE", "DESCRIPTION", "NEXT_AGENT_ID"])
        
        decision = response_dict.get('DECISION', '').lower().strip()
        if decision == "delegate":
            decision = 'forward'
        next_agent_id = None        # 转发的下一个Agent ID
        executable_tasks = None     # split之后目前Agent可以进行的任务
        description = None          # execute的时候对任务的思索描述

        if decision in ['forward']:
            
            try:
                next_agent_id = extract_agent_id(str(response_dict.get('NEXT_AGENT_ID', None)))
                
                if next_agent_id == None:
                    next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)
                else:
                    if next_agent_id in neighbors_info.keys():
                        next_agent_id_info = neighbors_info[next_agent_id]

                        ability = get_average_abilities_from_task_type(task.task_type, next_agent_id_info['abilities'])

                        if (ability >= 0.5 and
                            next_agent_id_info['current_load'] < 3 and
                            (next_agent_id_info['is_outgoing'] or next_agent_id_info['is_incoming'])):
                            next_agent_id = next_agent_id
                        else:
                            next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)
                    else:
                        next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)

            except Exception as e:
                next_agent_id = self.find_best_alternative_agent(task)

        if decision in ["split"]:
            executable_tasks = response_dict.get('EXECUTABLE', '').lower().strip()

        if decision in ["execute"]:
            description = response_dict.get('DESCRIPTION', '').lower().strip()
            
        execution_time = time.time() - start_time

        router_experience=RouterExperience(
            decision=decision,

            task_id=task.task_id,
            task_type=task.task_type,
            task_major_problem=task.major_problem,
            task_completed_work=copy.deepcopy(task.completed_work),
            task_remaining_work=task.remaining_work,

            initial_agent_id=0,                
            source_agent_id=self.agent_id,     
            target_agent_id=next_agent_id,      

            route_history=task_history,       
            execution_time=execution_time,
            success=False,               
            timestamp=time.time()
        )

        output_dict = {
            "response": response,
            "response_dict": response_dict,
            "query_prompt":query_prompt,

            "decision": decision,
            "next_agent_id": next_agent_id,
            "executable_tasks": executable_tasks,
            "description": description,
            "execution_time": execution_time,
            "experience": router_experience,
        }

        return output_dict


    def router_decide_next_agent_id(self, task, self_info, neighbors_info, task_history, constraints):
        start_time = time.time()
    
        examples = [] if is_scripted_agent(self.agent_id, self.llm) else self.router_pool.get_relevant_experiences(task, self.agent_id, self.retrieval_num)

        query_prompt = self.router_make_prompt_for_next_agent_id(
            experiences=examples,
            task=task, 
            prompt_template=ROUTER_PROMPT_DECIDE_NEXT_AGENT_ID_FORMAT, 
            self_info=self_info, 
            neighbors_info=neighbors_info,
            constraints=constraints
            )

        
        logger.debug(f"query_prompt for router_decide_next_agent_id: {query_prompt}")

        response = self.get_llm_response(
            system_prompt=ROUTER_DECIDE_NEXT_AGENT_ID_SYSTEM_PROMPT, 
            query_prompt=query_prompt
            )
        

        response_dict = extract_content_as_dict(long_string=response, short_strings=["DECISION", "REASON", "NEXT_AGENT_ID"])
        decision = response_dict.get('DECISION', '').lower().strip()
        reason = response_dict.get('REASON', '').lower().strip()
        next_agent_id = None

        if decision in ["completed"]:
            pass
        elif decision in ["incompleted"]:
            try:
                next_agent_id = extract_agent_id(str(response_dict.get('NEXT_AGENT_ID', None)))
                
                if next_agent_id == None:
                    next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)
                else:
                    if next_agent_id in neighbors_info.keys():
                        next_agent_id_info = neighbors_info[next_agent_id]
                        ability = get_average_abilities_from_task_type(task.task_type, next_agent_id_info['abilities'])
                        if (ability >= 0.5 and
                            next_agent_id_info['current_load'] < 3 and
                            (next_agent_id_info['is_outgoing'] or next_agent_id_info['is_incoming'])):
                            next_agent_id = next_agent_id
                        else:
                            next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)
                    else:
                        next_agent_id = self.find_best_alternative_agent(task.task_type, neighbors_info)

            except ValueError:
                next_agent_id = self.find_best_alternative_agent(task)



        execution_time = time.time() - start_time
        router_experience=RouterExperience(
            decision=decision,

            task_id=task.task_id,
            task_type=task.task_type,
            task_major_problem=task.major_problem,
            task_completed_work=copy.deepcopy(task.completed_work),
            task_remaining_work=task.remaining_work,

            initial_agent_id=0,               
            source_agent_id=self.agent_id,     
            target_agent_id=next_agent_id,     

            route_history=task_history,             
            execution_time=execution_time,
            success=False,               
            timestamp=time.time()
        )

        output_dict = {
            "response": response,
            "response_dict": response_dict,
            "query_prompt": query_prompt,
            "decision": decision,
            "reason": reason,
            "next_agent_id": next_agent_id,
            "execution_time": execution_time,
            "experience": router_experience,
        }


        return output_dict



    def router_make_prompt(self, experiences, task, decide_mode, self_info, neighbors_info, share_memories=False):
        current_task_type = task.task_type
        if decide_mode == "Normal":
            prompt_template = ROUTER_PROMPT_FORMAT
        elif decide_mode == "Split_and_Execute":
            prompt_template = ROUTER_PROMPT_SPLIT_AND_EXECUTE_FORMAT
        
        experience_strs = []
        for exp in experiences:
            experience_str = ""
            experience_str += f"- Task ID: {exp.task_id}\n"
            experience_str += f"- Task Type: {exp.task_type}\n"
            experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
            experience_str += f"- Task Progress: {render_completed_work(exp.task_completed_work)}\n"
            experience_str += f"- Task Decision: {exp.decision}\n"
            if exp.target_agent_id != None:
                experience_str += f"- Next Agent ID: {exp.target_agent_id}\n"
            experience_strs.append(experience_str)
        
        if len(experience_strs) == 0:
            total_experience_str = "None experience yet."
        else:
            total_experience_str = "\n".join(experience_strs)
            
        
        agent_info = "\nAvailable Agents:\n"
        if share_memories:
            for neighbors_id, single_neighbors_info in neighbors_info.items():
                if decide_mode == "Normal" and task.in_forward_history(single_neighbors_info['agent_id']):
                    continue 
                
                processed_task_str = ""
                if len(single_neighbors_info['processed_tasks']) == 0:
                    processed_task_str = "None processed task yet."
                else:
                    for i, exp in enumerate(single_neighbors_info['processed_tasks']):
                        processed_task_str += "\n"
                        processed_task_str += f"- Example {i+1}\n"
                        processed_task_str += f"- Task Type: {exp.task_type}\n"
                        processed_task_str += f"- Task Major Task: {exp.task_major_problem}\n"
                        processed_task_str += f"- Task Progress: {render_completed_work(exp.task_completed_work)}\n"
                        processed_task_str += f"- Task Solution: {exp.result}\n"
                
                agent_info += f"\nAgent {single_neighbors_info['agent_id']}:"
                agent_info += f"\n- Abilities:"
                for ability_type, ability in single_neighbors_info['abilities'].items():
                    agent_info += f"\n  * {ability_type}: {ability:.2f}"
                agent_info += f"\n- Current Load: {single_neighbors_info['current_load']}"


                if float(single_neighbors_info['task_type_success_rate']) != 0:
                    agent_info += f"\n- Success Rate for This Task Type: {single_neighbors_info['task_type_success_rate']:.2%}"
                else:
                    agent_info += f"\n- Success Rate for This Task Type: Not Succeed Yet."
                agent_info += f"\n- Processed Tasks: {processed_task_str}"
                agent_info += "\n"
        else:
            for neighbors_id, single_neighbors_info in neighbors_info.items():
                agent_info += f"\nAgent {single_neighbors_info['agent_id']}:"
                for ability_type, ability in single_neighbors_info['abilities'].items():
                    agent_info += f"\n  * {ability_type}: {ability:.2f}"
                    
            agent_info += "\n"

        task_type_success_rate = float(self_info['success_rate'][current_task_type])
        if task_type_success_rate == 0:
            task_type_success_rate_str = "Not Succeed Yet"
        else:
            task_type_success_rate_str = f"{task_type_success_rate:.2%}"

        current_agent_info = ""
        current_agent_info += f"- Agent ID: {self_info['agent_id']}\n"
        current_agent_info += f"- Ability for this task type: \n"
        for ability_type in task_to_ability_map[current_task_type]:
            current_agent_info += f"\t {ability_type}: {self_info['abilities'][ability_type]}\n"
        current_agent_info += f"- Success rate for this task type: {task_type_success_rate_str}\n"

        if not task.completed_work:
            progress_text = "None progress yet."
        else:
            progress_text = render_completed_work(task.completed_work)


        query_prompt = prompt_template.format(
            major_problem=with_private_info(task, self.agent_id),
            experiences=total_experience_str,
            task_context=progress_text,
            current_agent_info=current_agent_info,
            task_type=current_task_type,
            task_description=task.remaining_work if task.remaining_work != "" else task.major_problem,
            agent_info=agent_info,
            )
        

        return query_prompt


    def router_make_prompt_for_next_agent_id(self, experiences, task, prompt_template, self_info, neighbors_info, constraints):
        current_task_type = task.task_type
        experience_strs = []
        for exp in experiences:
            experience_str = ""
            experience_str += f"- Task ID: {exp.task_id}\n"
            experience_str += f"- Task Type: {exp.task_type}\n"
            experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
            experience_str += f"- Task Progress: {render_completed_work(exp.task_completed_work)}\n"
            experience_str += f"- Task Decision: {exp.decision}\n"
            if exp.target_agent_id != None:
                experience_str += f"- Next Agent ID: {exp.target_agent_id}\n"
            experience_strs.append(experience_str)

        if len(experience_strs) == 0:
            total_experience_str = "None experience yet."
        else:
            total_experience_str = "\n".join(experience_strs)

        agent_info = "\nAvailable Agents:\n"

        for neighbors_id, single_neighbors_info in neighbors_info.items():
            if task.in_forward_history(single_neighbors_info['agent_id']):
                continue 
            
            processed_task_str = ""
            for i, exp in enumerate(single_neighbors_info['processed_tasks']):
                processed_task_str += "\n"
                processed_task_str += f"- Example {i+1}\n"
                processed_task_str += f"- Task Type: {exp.task_type}\n"
                processed_task_str += f"- Task Major Task: {exp.task_major_problem}\n"
                processed_task_str += f"- Task Progress: {render_completed_work(exp.task_completed_work)}\n"
                processed_task_str += f"- Task Solution: {exp.result}\n"
            
            agent_info += f"\nAgent {single_neighbors_info['agent_id']}:"
            agent_info += f"\n- Abilities:"
            for ability_type, ability in single_neighbors_info['abilities'].items():
                agent_info += f"\n  * {ability_type}: {ability:.2f}"
            agent_info += f"\n- Current Load: {single_neighbors_info['current_load']}"
            if float(single_neighbors_info['task_type_success_rate']) != 0:
                agent_info += f"\n- Success Rate for This Task Type: {single_neighbors_info['task_type_success_rate']:.2%}"
            else:
                agent_info += f"\n- Success Rate for This Task Type: Not Succeed Yet."
            agent_info += f"\n- Processed Tasks: {processed_task_str}"
            agent_info += "\n"


        if not task.completed_work:
            progress_text = "None progress yet."
        else:
            progress_text = render_completed_work(task.completed_work)

        task_type_success_rate = float(self_info['success_rate'][current_task_type])
        if task_type_success_rate == 0:
            task_type_success_rate_str = "Not Succeed Yet"
        else:
            task_type_success_rate_str = f"{task_type_success_rate:.2%}"

        current_agent_info = ""
        current_agent_info += f"- Agent ID: {self_info['agent_id']}\n"
        current_agent_info += f"- Ability for this task type: \n"
        for ability_type in task_to_ability_map[current_task_type]:
            current_agent_info += f"\t {ability_type}: {self_info['abilities'][ability_type]}\n"
        current_agent_info += f"- Success rate for this task type: {task_type_success_rate_str}\n"


        query_prompt = prompt_template.format(
            major_problem=with_private_info(task, self.agent_id),
            task_context=progress_text,
            experiences=total_experience_str,
            current_agent_info=current_agent_info,
            agent_info=agent_info
            )
        
        return query_prompt


    def find_best_alternative_agent(self, task_type, neighbors_info):
        candidates = []
        for agent_id, info in neighbors_info.items():
            if agent_id != self.agent_id:
                if not (info['is_incoming'] or info['is_outgoing']):
                    continue  
    
                ability = get_average_abilities_from_task_type(task_type, info['abilities'])
                load = info['current_load']
                success_rate = info['success_rate'][task_type]
                
                score = (
                    ability * 0.4 + 
                    (1 - load/3) * 0.3 +  
                    success_rate * 0.2 + 
                    (1.0 if info['is_outgoing'] else 0.5) * 0.1 
                )
                
                if ability >= 0.5 and load < 3:  
                    candidates.append((score, agent_id))
        
        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        else:
            logger.warning(f"No suitable agent found for task type {task_type}, falling back to a random neighbor")
            candidates = [agent_id for agent_id in neighbors_info.keys() if agent_id != self.agent_id]
            logger.debug(f"The neighbors are {candidates}")
            if not candidates:
                logger.warning(f"No neighbors found for agent {self.agent_id}")
                return None
            return random.choice(candidates)

    def get_newest_experiences(self, task_type, k=50):
        return self.router_pool.get_newest_experience(task_type, k)


    def add_experience_to_pool(self, agent_abilities, experience):
        self.router_pool.add_experience(agent_abilities, experience)
        return 0 


    def set_experience_success_state(self, task_id, task_type, success):
        self.router_pool.set_experience_success_state(task_id, task_type, success)


    def get_all_experiences(self):
        return self.router_pool.get_all_experiences()



class ExecutorModule:
    def __init__(self,  agent_id, memory_limit, llm, retrieval_num):
        self.agent_id = agent_id
        
        self.memory_limit = memory_limit
        self.executor_pool = ExecutorExperiencePool(capacity=memory_limit)
        self.llm = llm
        self.retrieval_num = retrieval_num
        self.get_response = timed_llm(run_trace.traced(
            manual_backend.task_override(agent_id, llm_model_response_map[self.llm]), agent_id), agent_id, self.llm)

    def get_relevant_experiences(self, task):
        if is_scripted_agent(self.agent_id, self.llm):
            return []
        return self.executor_pool.get_relevant_experiences(task,top_k=self.retrieval_num)

    def get_relevant_experiences_by_thought(self, task):
        if is_scripted_agent(self.agent_id, self.llm):
            return []
        return self.executor_pool.get_relevant_experiences_by_thought(task,top_k=self.retrieval_num)

    def generate_thought(self, task, experiences):
        query_prompt = self.executor_make_prompt_for_thought(
            experiences=experiences, 
            task=task, 
            prompt_template=SPLIT_THOUGHT_PROMPT_FORMAT)

        response = self.get_response(
            system_prompt=THOUGHT_SYSTEM_PROMPT, 
            query_prompt=query_prompt
            )

        response_dict = extract_content_as_dict(long_string=response, short_strings=["RESULT"])
        thought = response_dict.get("RESULT", response) 

        output = {
            "response": response,
            "response_dict": response_dict,
            "query_prompt": query_prompt,
            "thought": thought,
        }

        return output


    def executor_execute_task(self, abilities, task, experiences, constraints, execution_choice):
        start_time = time.time()
        if execution_choice == "execute":
            thought_query_prompt = self.executor_make_prompt_for_thought(experiences=experiences,
                                                                          task=task,
                                                                          prompt_template=EXECUTOR_THOUGHT_PROMPT_FORMAT)
            logger.debug(f"thought query prompt for execute: {thought_query_prompt}")
            thought_response = self.get_response(
                system_prompt=THOUGHT_SYSTEM_PROMPT,
                query_prompt=thought_query_prompt
                )

            thought_response_dict = extract_content_as_dict(long_string=thought_response, short_strings=["RESULT"])
            thought = thought_response_dict.get("RESULT", thought_response)

            logger.debug(f"thought before execute: {thought}")

            query_prompt = self.executor_make_prompt(experiences=experiences,
                                                     task=task,
                                                     prompt_template=EXECUTOR_PROMPT_FORMAT,
                                                     constraints=constraints,
                                                     execution_choice=execution_choice,
                                                     thought=thought)
            logger.debug(f"query prompt for execute: {query_prompt}")
        elif execution_choice == "split":
            thought = task.thought
            logger.debug(f"when making prompt, thought: {thought}")
            query_prompt = self.executor_make_prompt(experiences=experiences,
                                                    task=task, prompt_template=SPLIT_EXECUTOR_PROMPT_FORMAT,
                                                    constraints=constraints, execution_choice=execution_choice,
                                                    thought=thought)

        response = self.get_response(
            system_prompt=EXECUTOR_SYSTEM_PROMPT, 
            query_prompt=query_prompt
            )
        
        response_dict = extract_content_as_dict(long_string=response, short_strings=["RESULT"])

        result = response_dict.get("RESULT", response)

        execution_time = time.time() - start_time
        executor_experience=ExecutorExperience(
            task_id=task.task_id,
            task_type=task.task_type,
            task_major_problem=task.major_problem,
            task_remaining_work=task.remaining_work,
            task_completed_work=copy.deepcopy(task.completed_work),
            task_thought=task.thought,

            result=result,
            execution_time=execution_time, 
            success=False,                   
            error_type = None,              
            source_agent_id=self.agent_id,
            performance_metrics=None,      
            timestamp=time.time()
        )

        # self.executor_pool.add_experience(abilities, executor_experience)

        output = {
            "response": response,
            "response_dict": response_dict,
            "query_prompt": query_prompt,
            "result": result,
            "execution_time": execution_time,
            "experience": executor_experience,
        }

        return output
    
    def executor_make_prompt_for_thought(self, experiences, task, prompt_template):
        
        experience_strs = []
        for exp in experiences:
            experience_str = ""
            experience_str += f"- Task ID: {exp.task_id}\n"
            experience_str += f"- Task Type: {exp.task_type}\n"
            experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
            experience_str += f"- Task Progress: {render_completed_work(exp.task_completed_work)}\n"
            experience_str += f"- Task Description: {exp.task_remaining_work}\n"
            experience_str += f"- Task Thought: {exp.task_thought}\n"
            experience_str += f"- Task Result: {exp.result}\n"

        if len(experience_strs) == 0:
            total_experience_str = "None experience yet."
        else:
            total_experience_str = "\n".join(experience_strs)

        if not task.completed_work:
            progress_text = "None progress yet."
        else:
            progress_text = render_completed_work(task.completed_work)
        
        query_prompt = prompt_template.format(
            major_problem=with_private_info(task, self.agent_id),
            experiences = total_experience_str,
            task_context = progress_text,
            task_description = task.remaining_work
        )

        return query_prompt



    def executor_make_prompt(self, experiences, task, prompt_template, constraints, execution_choice, thought):
        experience_strs = []
        logger.debug(f"when making prompt, len(experiences): {len(experiences)}")
        if execution_choice == "execute":
            logger.debug(f"when making prompt, execution_choice: {execution_choice}")
            for exp in experiences:
                experience_str = ""
                experience_str += f"- Task ID: {exp.task_id}\n"
                experience_str += f"- Task Type: {exp.task_type}\n"
                experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
                experience_str += f"- Task Progress: {render_completed_work(exp.task_completed_work)}\n"
                experience_str += f"- Task Description: {exp.task_remaining_work}\n"
                experience_str += f"- Task Result: {exp.result}\n"
                experience_strs.append(experience_str)
        elif execution_choice == "split":
            logger.debug(f"when making prompt, execution_choice: {execution_choice}")
            for exp in experiences:
                if not exp.task_completed_work:
                    continue
                experience_str = ""
                experience_str += f"- Task ID: {exp.task_id}\n"
                experience_str += f"- Task Type: {exp.task_type}\n"
                experience_str += f"- Task Major Task: {exp.task_major_problem}\n"
                experience_str += f"- Task Progress: {render_completed_work(exp.task_completed_work)}\n"
                experience_strs.append(experience_str)

        if len(experience_strs) == 0:
            total_experience_str = "None experience yet."
        else:
            total_experience_str = "\n".join(experience_strs)


        if not task.completed_work:
            progress_text = "None progress yet."
        else:
            progress_text = render_completed_work(task.completed_work)
        

        query_prompt = prompt_template.format(
            major_problem=with_private_info(task, self.agent_id),
            experiences = total_experience_str,
            task_context = progress_text,
            task_description = task.remaining_work,
            thought = thought,
            constraints = constraints
        )

        return query_prompt


    def add_experience_to_pool(self, agent_abilities, experience):
        self.executor_pool.add_experience(agent_abilities, experience)
        return 0 

    def set_experience_success_state(self, task_id, task_type, success):
        self.executor_pool.set_experience_success_state(task_id, task_type, success)
        # self.smart_evivation
        return 0


    def get_all_experiences(self):
        return self.executor_pool.get_all_experiences()

class Agent:
    def __init__(self, agent_config, global_router_experience_flag):
        self.agent_config = agent_config
        self.agent_id = agent_config["id"]
        self.config = agent_config["config"]
        self.abilities = self.config["abilities"]
        self.llm = self.config["LLM"]

        self.executor_retrieval_num = self.config["executor_retrieval_num"]
        self.router_retrieval_num = self.config["router_retrieval_num"]

        self.collect_neighbors_info = None

        self._load_local = threading.local()
        self.task_correlations = {}
        
        self.success_rate = {}
        self.total_task_num = {}
        self.successful_task_num = {}
        for key, value in BASIC_PROBLEM_COMPLEXITY.items():
            self.success_rate[key] = 0
            self.total_task_num[key] = 0
            self.successful_task_num[key] = 0


        self.decay_rate = self.config["decay_rate"]
        self.decay_interval = self.config["decay_interval"]
        self.decay_count_down = {}

        self.router_module = RouterModule(agent_id=self.agent_id, memory_limit=self.config["router_memory_limit"], llm=self.llm, global_router_experience_flag=global_router_experience_flag, retrieval_num = self.router_retrieval_num)
        self.router_module.set_get_self_agent_info(self.get_self_info)

        self.executor_module = ExecutorModule(
            agent_id=self.agent_id, 
            memory_limit=self.config["executor_memory_limit"], 
            llm=self.llm,
            retrieval_num = self.executor_retrieval_num
            )


        #### PKI ####
        self.sk, self.pk = generate_ed25519_keypair()
        # qid -> trace
        self.most_recent_trace = {}
        #### PKI ####


    @property
    def current_load(self):
        return getattr(self._load_local, "value", 0)

    @current_load.setter
    def current_load(self, value):
        self._load_local.value = value

    def set_collect_neighbors_info(self, collect_neighbors_info):
        self.collect_neighbors_info = collect_neighbors_info



    def decide_action(self, task_chain, forward_times):
        current_task = task_chain.get_current_task()

        self_info = self.get_self_info()
        neighbors_info = self.collect_neighbors_info(agent_id=self.agent_id, task=current_task)

        decide_mode = "Normal"
        No_Neighbor_Flag = True
        for key, value in neighbors_info.items():
            if key not in current_task.forward_history:
                No_Neighbor_Flag = False
        if (No_Neighbor_Flag) or (forward_times <=0):
            decide_mode = "Split_and_Execute"
        
        output = self.router_module.router_decide_action(
            task=current_task, 
            self_info=self_info, 
            neighbors_info=neighbors_info,
            task_history=task_chain.task_history,
            decide_mode=decide_mode,
            )
        
        newest_experiences = self.router_module.get_newest_experiences(task_type=current_task.task_type, k=50)
        self.update_task_correlations(task_type=current_task.task_type, newest_experiences=newest_experiences)
        
        return output


    def decide_next_agent_id(self, task_chain,constraints):
        current_task = task_chain.get_current_task()

        self_info = self.get_self_info()
        neighbors_info = self.collect_neighbors_info(agent_id = self.agent_id, task=current_task)

        output = self.router_module.router_decide_next_agent_id(
            task=current_task, 
            self_info=self_info, 
            neighbors_info=neighbors_info,
            task_history=task_chain.task_history,
            constraints=constraints,
            )
        
        newest_experiences = self.router_module.get_newest_experiences(task_type=current_task.task_type, k=50)
        self.update_task_correlations(task_type=current_task.task_type, newest_experiences=newest_experiences)

        return output



    def execute_task(self, task_chain, constraints, split_constraints, thought_constraints, use_react=True, decision="execute"):
        current_task = task_chain.get_current_task()
        logger.info(f"Agent {self.agent_id} is Executing Task {current_task.task_id}")

        self.current_load += 1
        self.total_task_num[current_task.task_type] += 1

        relevant_experiences = self.executor_module.get_relevant_experiences(current_task)

        if use_react == True:

            output = self.executor_module.generate_thought(current_task, relevant_experiences)
            current_task.set_thought(output["thought"])
            
            relevant_experiences_by_thought = self.executor_module.get_relevant_experiences_by_thought(current_task)
            relevant_experiences = relevant_experiences_by_thought
        
        logger.debug(f"when execute task, len(relevant_experiences): {len(relevant_experiences)}")

        if decision == "execute":
            output = self.executor_module.executor_execute_task(abilities=self.abilities, task=current_task, experiences=relevant_experiences, constraints=constraints, execution_choice="execute")
        elif decision == "split":   
            output = self.executor_module.executor_execute_task(abilities=self.abilities, task=current_task, experiences=relevant_experiences, constraints=split_constraints, execution_choice="split")

        self.current_load -= 1
        if current_task.task_type not in self.decay_count_down.keys():
            self.decay_count_down[current_task.task_type] = self.decay_interval
        return output


    def get_relevant_experence(self, task):
        return self.executor_module.executor_pool.get_relevant_experiences(task, top_k=self.executor_retrieval_num)


    def get_self_info(self):
        self_info = {
            "agent_id": self.agent_id,
            "current_load": self.current_load,
            "success_rate": self.success_rate, 
            "abilities": manual_backend.effective_abilities(self.agent_id, self.abilities),
        }
        return self_info


    def add_executor_experience(self, experience):
        self.executor_module.add_experience_to_pool(self.abilities, experience)
        

    def add_router_experience(self, experience):
        self.router_module.add_experience_to_pool(self.abilities, experience)
    


    def get_all_experiences(self):
        router_experiences = self.router_module.get_all_experiences()
        executor_experiences = self.executor_module.get_all_experiences()

        return {
            "router_experiences": router_experiences,
            "executor_experiences": executor_experiences,
        }

    
    def update_abilities(self, task_type, execution_time, success):

        if success:  
            self.successful_task_num[task_type] += 1
            self.success_rate[task_type] = self.successful_task_num[task_type] / self.total_task_num[task_type]

            ability_gain = 0.1  

            for ability_type in task_to_ability_map[task_type]:
                self.abilities[ability_type] = min(2.0, self.abilities[ability_type] + ability_gain)  

            for related_type, correlation in self.task_correlations[task_type].items():
                if correlation > 0.3: 
                    for ability_type in task_to_ability_map[related_type]:
                        self.abilities[ability_type] = min(2.0, self.abilities[ability_type] + ability_gain * correlation * 0.5)

        else:
            pass



    def update_task_correlations(self, task_type, newest_experiences) -> None:
        if task_type not in self.task_correlations:
            self.task_correlations[task_type] = {}
        
        for experience in newest_experiences:
            if experience.task_type != task_type:
                correlation = self.calculate_correlation(newest_experiences, task_type, experience.task_type)
                current_correlation = self.task_correlations[task_type].get(experience.task_type, 0)
                self.task_correlations[task_type][experience.task_type] = current_correlation * 0.9 + correlation * 0.1
                    



    def calculate_correlation(self, newest_history, task_type1, task_type2):
        task_type1_count = sum(1 for entry in newest_history if entry['task_type'] == task_type1)
        task_type2_count = sum(1 for entry in newest_history if entry['task_type'] == task_type2)
        
        if task_type1_count == 0 or task_type2_count == 0:
            return 0.0
            
        cooccurrence = 0
        for i in range(len(newest_history)-1):
            if newest_history[i]['task_type'] == task_type1 and newest_history[i+1]['task_type'] == task_type2:
                cooccurrence += 1
                
        correlation = cooccurrence / min(task_type1_count, task_type2_count)
        return correlation



    def set_executor_pool_result(self, task_id, task_type, success):
        for key, value in self.decay_count_down.items():
            self.decay_count_down[key] = value - 1
        if success:
            self.decay_count_down[task_type] = self.decay_interval

        for key, value in self.decay_count_down.items():
            if value <= 0:
                self.decay_abilities(task_type=key)
                self.decay_count_down[key] = self.decay_interval
            
        self.executor_module.set_experience_success_state(task_id, task_type, success)



    def set_router_pool_result(self, task_id, task_type, success):
        self.router_module.set_experience_success_state(task_id, task_type, success)

    
    def get_pool_info(self):
        return (self.router_module.router_pool,self.executor_module.executor_pool)
    

    def decay_abilities(self, task_type) -> None:
        for ability_type in task_to_ability_map[task_type]:
            self.abilities[ability_type] = max(0.1, self.abilities[ability_type] * (1 - self.decay_rate))


import yaml
import logging
import os
import re
import time
import json
import pickle
from config.setting import OPENAI_API_KEY, OPENAI_API_BASE, AGENT_MODEL
from config.setting import task_to_ability_map
import numpy as np
from . import run_trace
logger = logging.getLogger(__name__)

MAX_RETRIES = 3



def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, file_path):
    with open(file_path, 'w') as w:
        json.dump(data, w, indent=4)

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_yaml(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def get_gpt_response(system_prompt, query_prompt, model=AGENT_MODEL):
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        max_retries=0,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=query_prompt,
                max_output_tokens=2048,
                reasoning={"effort": "low"},
            )

            content = response.output_text

            usage = getattr(response, "usage", None)
            run_trace.record_api_usage(
                system_prompt, query_prompt, content,
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0)

            if content is None:
                print("DEBUG empty output, status =", response.status)
                print("DEBUG response =", response)
                return ""

            return content.strip()

        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(1)
            

def get_qwen_response(system_prompt, query_prompt, model=AGENT_MODEL):
    from openai import OpenAI

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_API_BASE,
        max_retries=0,
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=query_prompt,
                max_output_tokens=8192,
                reasoning={"effort": "low"},
            )

            content = response.output_text or ""

            usage = getattr(response, "usage", None)
            run_trace.record_api_usage(
                system_prompt, query_prompt, content,
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0)

            if content is None:
                print("DEBUG empty output, status =", response.status)
                print("DEBUG response =", response)
                return ""

            if response.status == "incomplete" or not content.strip():
                print("DEBUG truncated/empty:", response.status,
                    getattr(response, "incomplete_details", None))
                continue  # retry

            return content.strip()

        except Exception as e:
            print(f"API call attempt {attempt + 1} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(1)

import re

def parse_decision_text(decision_text):
    """
    Parse a decision text into a dict of UPPERCASE keys and their values.
    Each key line looks like 'KEY:'; subsequent lines until next KEY belong to the value.
    """
    output_dict = {}
    current_key = None
    current_value = []

    lines = decision_text.strip().split('\n')

    for line in lines:
        line = line.strip()

        if re.match(r'^[A-Z_]+:', line):
            if current_key is not None:
                output_dict[current_key] = ' '.join(current_value).strip()

            key, value = line.split(':', 1)
            current_key = key.strip()
            current_value = [value.strip()] if value.strip() else []
        else:
            # Continuation of current value
            if current_key is not None:
                current_value.append(line)

    # Save the last pair
    if current_key is not None:
        output_dict[current_key] = ' '.join(current_value).strip()

    return output_dict


def parse_string_to_dict(text):
    result = {}
    lines = text.strip().split("\n")

    for line in lines:
        match = re.match(r"\b[A-Z_]{2,}\b", line)
        if match:
            key = match.group(0)
            value = line[match.end():].strip() 
            value = re.sub(r"^[\s:]+|[\s]+$", "", value)
            result[key] = value
        
    return result


def extract_content_as_dict(long_string, short_strings):

    pattern = "|".join(re.escape(short_string) for short_string in short_strings)
    regex = re.compile(f"({pattern})")

    matches = list(regex.finditer(long_string))
    result = {}

    for i, match in enumerate(matches):
        key = match.group() 
        start = match.end()  
        
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(long_string)  

        value = long_string[start:end].strip().strip(":")
        result[key] = value

    return result




def extract_agent_id(text):
    try:
        bracket_match = re.search(r'\[(.*?)\]', text)
        if bracket_match:
            text = bracket_match.group(1).strip()

        agent_match = re.search(r'(Agent\s*(\d+)|(\d+))', text, re.IGNORECASE)
        if agent_match:
            agent_id = agent_match.group(2) if agent_match.group(2) else agent_match.group(3)
            return int(agent_id.strip())

        return int(text.strip())

    except (ValueError, TypeError) as e:
        print(f"Could not extract agent ID from: {text}")
        logger.info(f"Could not extract agent ID from: {text}")
        return None






def parse_split_pattern_to_subtask(split_text):
    if not split_text:
        return None
        
    try:
        re_pattern = r'(\b(?:\d+\.\s+|Subtask\s+\d+)\s*[^;]+(?:;|$))'
        tasks = re.findall(re_pattern, split_text)
        tasks = [task.strip() for task in tasks]
        difficulty = [estimate_subtask_difficulty(task) for task in tasks]
        required_capabilities = [analyze_required_capabilities(task) for task in tasks]


        return tasks, difficulty, required_capabilities
    
    except Exception as e:
        print(f"Error when parsing text: {split_text}")
        print(f"Error parsing split pattern: {e}")
        return None



def estimate_subtask_difficulty(description):
        base_difficulty = len(description.split()) / 10
        if any(kw in description.lower() for kw in ['complex', 'difficult', 'advanced']):
            base_difficulty *= 1.5
        return min(5.0, base_difficulty)


def analyze_required_capabilities(description):
    capabilities = []
    keywords = {
        'math': ['calculate', 'solve', 'equation'],
        'coding': ['function', 'code', 'implement'],
        'language': ['translate', 'write', 'summarize']
    }
    
    description = description.lower()
    for capability, words in keywords.items():
        if any(word in description for word in words):
            capabilities.append(capability)
            
    return capabilities
    

def extract_result(text):
    match = re.search(r'RESULT:\s*(\d+)', text)
    if match:
        return int(match.group(1))  
    else:
        return None  



def calculate_cos_similarity_A_and_Batch_B(A, B):
    dot_product = np.dot(A, B.T)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B, axis=1)  # (4,)
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim


def calculate_cos_similarity_A_and_B(A, B):
    dot_product = np.dot(B, A)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    cosine_sim = dot_product / (norm_B * norm_A)
    return cosine_sim




def get_average_abilities_from_task_type(task_type, abilities_dict):
    ability = 0
    count = 0
    for ability_type in task_to_ability_map[task_type]:
        count += 1
        ability += abilities_dict.get(ability_type, 0)
    if count != 0:
        ability /= count
        return ability
    else:
        return 0



def extract_all_results(text):
    result_pattern = re.compile(r"(?:Result:|RESULT:)\s*(.*?)(?=\n|$)", re.DOTALL)
    results = re.findall(result_pattern, text)
    return results

def extract_answer(text):
    # Extract all answers after "Result"
    results = extract_all_results(text)
    
    # Return the last answer (assuming the last one is the correct answer)
    if results:
        last_result = results[-1].strip()
        # Check if the last result is not empty
        if last_result:
            return last_result
    return text

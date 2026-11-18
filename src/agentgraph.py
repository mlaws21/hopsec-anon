import random
import networkx as nx
import copy
import logging

from .agent import Agent, is_scripted_agent
from config.setting import task_to_ability_map
from .vtrace import agent_name


class AgentGraph:
    def __init__(self, agent_config, agent_graph_config):
        self.agent_config = agent_config
        self.agent_graph_config = agent_graph_config
        self.edge_weight = {}
        self.edge_success_rate = {}

        # Edge-weight update knobs (optional agent_graph_config keys)
        # defaults are what agentnet uses
        # added so that in qwen experiements where task success is lower agents don't isolate themselves
        self.edge_success_factor = agent_graph_config.get("edge_success_factor", 1.1)
        self.edge_failure_factor = agent_graph_config.get("edge_failure_factor", 0.9)
        self.edge_time_scale = agent_graph_config.get("edge_time_scale", 0.1) # 0 disables the time penalty
        self.edge_weight_min = agent_graph_config.get("edge_weight_min", 0.1)
        self.edge_weight_max = agent_graph_config.get("edge_weight_max", 2.0)
        self.edge_prune_threshold = agent_graph_config.get("edge_prune_threshold", 0.3)
        self.edge_initial_weight = agent_graph_config.get("edge_initial_weight", 1.0)

        self.agents = self.initilize_agents(agent_config, agent_graph_config["global_router_experience"])
        self.agent_neighbor_dict = self.initilize_graph(agent_graph_config)
        
        
        #### PKI ####
        self.pks = {agent_name(aid): a.pk for aid, a in self.agents.items()}
        #### PKI ####


    def initilize_agents(self, agent_config, global_router_experience_flag):
        agents = {}
        for single_agent_config in agent_config:
            single_agent_config_copy = copy.deepcopy(single_agent_config)
            agents[single_agent_config_copy["id"]] = Agent(single_agent_config_copy, global_router_experience_flag)

        for agent_id, agent in agents.items():
            agent.set_collect_neighbors_info(self.collect_neighbors_info)

        for source_agent_id, agent in agents.items():
            if source_agent_id not in self.edge_weight.keys():
                self.edge_weight[source_agent_id] = {}
                self.edge_success_rate[source_agent_id] = {}
            for target_agent_id, agent in agents.items():
                if source_agent_id != target_agent_id:
                    self.edge_weight[source_agent_id][target_agent_id] = self.edge_initial_weight
                    self.edge_success_rate[source_agent_id][target_agent_id] = 0.0

        return agents


    def initilize_graph(self, agent_graph_config):
        agent_neighbor_dict = {

        }
        if agent_graph_config["graph_type"] == "complete":
            for source_agent_id, source_agent in self.agents.items():
                if source_agent_id not in agent_neighbor_dict.keys():
                    agent_neighbor_dict[source_agent_id] = {
                        "incoming_agent_id": [],
                        "outcoming_agent_id": [],
                    }
                for target_agent_id, target_agent in self.agents.items():
                    if source_agent_id == target_agent_id:
                        continue
                    else:
                        agent_neighbor_dict[source_agent_id]["incoming_agent_id"].append(target_agent_id)
                        agent_neighbor_dict[source_agent_id]["outcoming_agent_id"].append(target_agent_id)
        
        return agent_neighbor_dict

    def sample_an_agent(self):
        agent_id = random.choice(list(self.agents.keys())) 
        return agent_id


    def select_an_agent(self, task_type):
        outcoming_agents_id = self.agents.keys()
        logging.debug(f"outcoming_agents_id = {outcoming_agents_id}")
        neighbors_info = {}
        
        for agent_id in outcoming_agents_id:
            agent = self.agents[agent_id]
            
            agent_info = agent.get_self_info()  
            success_rate = agent_info["success_rate"]
            abilities = agent_info["abilities"]

            if task_type not in success_rate:
                continue
        
            ability_names = task_to_ability_map[task_type]
            total_value, ability_num = 0, 0
            for name in ability_names:
                ability_num += 1
                total_value += abilities[name]
            average_ability_value = total_value / ability_num

            neighbors_info[agent_id] = {
                "agent_info": agent_info,
                "average_ability_value": average_ability_value
            }
            logging.debug(f"neighbors_info[agent_id]: {agent_id}, agent_info: {agent_info}, average_ability_value: {average_ability_value}")

        if not neighbors_info:
            logging.info(f"No suitable agent found to execute task {task_type}, randomly selecting an agent")
            return self.sample_an_agent()
            
        # Get all maximum average_ability_values
        max_value = max(neighbors_info[agent_id]["average_ability_value"] for agent_id in neighbors_info)
        
        # Filter out all agents with maximum value
        best_agents = [agent_id for agent_id, info in neighbors_info.items() if info["average_ability_value"] == max_value]
        
        # If there are multiple agents, randomly select one
        best_agent_id = random.choice(best_agents)

        logging.info(f"Selected agent {best_agent_id} to execute task {task_type}, average ability value: {neighbors_info[best_agent_id]['average_ability_value']}")
        
        return best_agent_id



    def collect_neighbors_info(self, agent_id, task):
        outcoming_neighbors_id  = self.agent_neighbor_dict[agent_id]["outcoming_agent_id"]

        calling_agent_is_manual = is_scripted_agent(agent_id, self.agents[agent_id].llm)
        neighbors_info = {}
        for neighbor_id in outcoming_neighbors_id:
            if self.edge_weight[agent_id][neighbor_id] <= self.edge_prune_threshold:
                # If edge weight is too low, discard this edge and no longer accept it when passing Agent neighbor information
                continue
            neighbor_agent = self.agents[neighbor_id]
            neighbor_agent_info = neighbor_agent.get_self_info()

            neighbor_agent_info["processed_tasks"] = [] #if calling_agent_is_manual else neighbor_agent.get_relevant_experence(task)
            neighbor_agent_info["success_rate"] = neighbor_agent_info["success_rate"]
            neighbor_agent_info["task_type_success_rate"] = neighbor_agent_info["success_rate"][task.task_type]
            neighbor_agent_info["is_incoming"] = False
            neighbor_agent_info["is_outgoing"] = True
            neighbors_info[neighbor_id]= neighbor_agent_info
        return neighbors_info


    def update_edge_weight(self, source_agent_id, target_agent_id, execution_time, success):
        
        current_rate = self.edge_success_rate[source_agent_id].get(target_agent_id, 0.6)
        self.edge_success_rate[source_agent_id][target_agent_id] = current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
            
        # Update edge weights
        current_weight = self.edge_weight[source_agent_id].get(target_agent_id, self.edge_initial_weight)
        success_factor = self.edge_success_factor if success else self.edge_failure_factor
        if execution_time > 0 and self.edge_time_scale > 0:
            time_factor = min(1.0, 1.0 / (execution_time * self.edge_time_scale))
        else:
            time_factor = 1.0

        logging.debug(f"Update edge weight: {source_agent_id} -> {target_agent_id}, success rate: {current_rate}, weight: {current_weight}, success factor: {success_factor}, time factor: {time_factor}")

        new_weight = current_weight * success_factor * time_factor

        # Ensure weights are within reasonable range
        self.edge_weight[source_agent_id][target_agent_id] = max(self.edge_weight_min, min(self.edge_weight_max, new_weight))
        logging.debug(f"Update edge weight: {source_agent_id} -> {target_agent_id}, weight: {self.edge_weight[source_agent_id][target_agent_id]}")
        logging.debug(f"Neighbor list: {self.agent_neighbor_dict[source_agent_id]['outcoming_agent_id']}")
        if self.edge_weight[source_agent_id][target_agent_id] <= self.edge_prune_threshold and self.agent_neighbor_dict[source_agent_id]["outcoming_agent_id"].count(target_agent_id) > 0:
            # Delete directly if weight is too small
            self.agent_neighbor_dict[source_agent_id]["outcoming_agent_id"].remove(target_agent_id)

    def get_all_agent_info(self):
        all_agent_info = { }
        for agent_id, agent in self.agents.items():
            all_agent_info[agent_id] = agent.get_self_info()
        return all_agent_info


    def get_all_agent_experiences(self):
        all_agent_experiences = { }
        for agent_id, agent in self.agents.items():
            all_agent_experiences[agent_id] = agent.get_all_experiences()
        return all_agent_experiences

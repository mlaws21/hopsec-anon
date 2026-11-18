import copy
import secrets
import time

from .vtrace import (progress_from_vet, work_refs, received_remaining_work,
                     major_problem_from_vet, final_result_from_vet)



def render_completed_work(completed_work):
    progress_text = ""
    for progress_dict in completed_work:
        progress_text += f"Agent {progress_dict['agent_id']} executed following tasks\n"
        progress_text += f"{progress_dict['description']}\n"
        progress_text += f"Agent {progress_dict['agent_id']} output the following answers\n"
        progress_text += f"{progress_dict['result']}\n"
    return progress_text



class Task:

    def __init__(self,
                 task_id,
                 task_type,
                 major_problem="",
                 completed_work=None,
                 remaining_work="",
                 thought="",
                 result="",
                 correct_answer="",
                 ):

        self.task_id: str = task_id
        self.task_type = task_type

        self.major_problem = major_problem

        self.completed_work = completed_work if completed_work is not None else []
        self.remaining_work = remaining_work
        self.thought = thought


        self.result = result

        self.forward_history = []
        self.correct_answer = correct_answer

    def __str__(self):
        return f"Task {self.task_id} is a {self.task_type} problem. Its major problem is '{self.major_problem}'. Its remaining work is {self.remaining_work}"

    def set_result(self, result):
        self.result = result

    def set_thought(self, thought):
        self.thought = thought

    def add_forward_history(self, agent_id):
        self.forward_history.append(agent_id)

    def in_forward_history(self, agent_id):
        return agent_id in self.forward_history



class TaskChain:
    def __init__(self, task: Task):
        self.original_task = task
        self.task_chain_id = task.task_id
        self.task_id_count = 1

        self.vet = []
        self._pending_work = None
        self.agent_id_of = {}

        self.task_type = task.task_type

        self._major_problem = task.major_problem

        self.current_task = task
        self.task_history = []
        self.task_chain = []

        self.private_info = getattr(task, "private_info", None)
        self.user_id = getattr(task, "user_id", None)
        self.deadline = time.time() + getattr(task, "timelimit", 600)
        self.allowed_agents = getattr(task, "allowed_agents", None)

        self.qid = str(secrets.randbits(64))
        
        self.was_tampered_all = None
        self.was_tampered_one = None

    @property
    def progress_list(self):
        return progress_from_vet(self.vet, self._pending_work,
                                 lambda name: self.agent_id_of.get(name, name))

    def progress_work_refs(self):
        return work_refs(self.vet, self._pending_work)

    def delete_progress(self, indices):
        indices = set(indices)
        targets = {id(ref) for i, ref in enumerate(self.progress_work_refs()) if i in indices}
        if self._pending_work is not None and id(self._pending_work[1]) in targets:
            self._pending_work = None
        for m, _ in self.vet:
            if m["sender_work"] is not None and id(m["sender_work"]) in targets:
                m["sender_work"] = None

    @property
    def major_problem(self):
        return major_problem_from_vet(self.vet) if self.vet else self._major_problem

    @major_problem.setter
    def major_problem(self, value):
        if self.vet:
            self.vet[0][0]["remaining_work"] = value
        else:
            self._major_problem = value

    def pop_pending_work(self):
        pending, self._pending_work = self._pending_work, None
        return pending[1] if pending is not None else None

    def receive(self):
        self.current_task.remaining_work = received_remaining_work(self.vet)


    def create_next_task(self):
        self.task_id_count += 1

        next_task = Task(
            task_id=f"{self.task_chain_id}_{self.task_id_count}",
            task_type=self.task_type,
            major_problem=self.major_problem,
            completed_work=copy.deepcopy(self.progress_list),
        )

        next_task.private_info = self.private_info

        self.task_chain.append(copy.deepcopy(self.current_task))
        self.current_task = next_task



    def render_progress_text(self):
        return render_completed_work(self.progress_list)


    def set_current_task_remaining_work(self, remaining_work):
        self.current_task.remaining_work = remaining_work


    def set_current_task_result(self, agent_id, result):
        self._pending_work = (agent_id, {
            "step_id": self.task_id_count,
            "description": self.current_task.remaining_work,
            "result": result
        })

        self.current_task.result = result


    def add_task_history(self,
                         format_response,
                         original_response,
                         experience,
                         agent_id,
                         execution_time,
                         mode):

        task_history = TaskHistory(
            task=copy.deepcopy(self.current_task),
            format_response=format_response,
            original_response=original_response,
            experience=experience,
            current_agent_id=agent_id,
            execution_time=execution_time,
            mode=mode
        )
        self.task_history.append(task_history)

    def get_current_task(self):
        return self.current_task

    def get_current_task_id(self):
        return self.current_task.task_id

    def get_final_result(self):
        return final_result_from_vet(self.vet)



class TaskHistory:
    def __init__(self,
                 task,
                 format_response,
                 experience,
                 original_response,
                 current_agent_id,
                 execution_time,
                 mode,
                 ):

        self.task = copy.deepcopy(task)

        self.format_response = format_response
        self.current_agent_id = current_agent_id
        self.original_response = original_response
        self.experience = experience
        self.execution_time = execution_time
        self.timestamp = time.time()
        self.mode=mode

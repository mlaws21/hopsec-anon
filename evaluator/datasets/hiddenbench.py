import os
import sys
import string
from typing import List, Optional

import pandas as pd

from evaluator.datasets.base_dataset import BaseDataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.task import Task

HF_PATH = "hf://datasets/YuxuanLi1225/HiddenBench/benchmark.json"
LETTERS = string.ascii_uppercase


class HiddenBenchDataset(BaseDataset):
    def __init__(self, num_agents: int, local_path: Optional[str] = None,
                 task_type: str = "hidden_profile") -> None:
        """
        num_agents: hidden facts are spread round-robin across this many agents.
        local_path: optional local copy of benchmark.json otherwise fetch over internet.
        task_type: standard since all tasks are similar.
        """
        self.num_agents = num_agents
        self.task_type = task_type
        self._total_df = self._load_data(local_path or HF_PATH)

    @staticmethod
    def get_domain() -> str:
        return 'hiddenbench'

    @staticmethod
    def _load_data(path: str) -> pd.DataFrame:
        df = pd.read_json(path)
        print(f"HiddenBench: {len(df)} tasks")
        return df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self._total_df)

    def __getitem__(self, index: int) -> pd.Series:
        return self._total_df.iloc[index]

    def postprocess_answer(self, answer) -> str:
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        return str(answer)

    @staticmethod
    def record_to_target_answer(record) -> str:
        """Correct answer as "(X)"."""
        options = list(record["possible_answers"])
        answer = str(record["correct_answer"]).strip().lower()
        for i, opt in enumerate(options):
            if answer == str(opt).strip().lower():
                return f"({LETTERS[i]})"
        raise ValueError(f"Task {record['id']}: answer {record['correct_answer']!r} not in {options}")

    def _record_to_task(self, record) -> Task:
        options = "\n".join(f"({LETTERS[i]}) {opt}" for i, opt in enumerate(record["possible_answers"]))
        shared = "\n".join(f"- {s}" for s in record["shared_information"])

        # Only shared facts go in the problem everyone sees.
        major_problem = (
            f"{record['description']}\n\n"
            f"Information:\n{shared}\n\n"
            f"Options:\n{options}"
        )

        task = Task(
            task_id=int(record["id"]),
            task_type=self.task_type,
            major_problem=major_problem,
            correct_answer=self.record_to_target_answer(record),
        )

        # private_info[i] = agent i's hidden facts (round-robin).
        # each agent must append its own entry to its prompt.
        hidden = list(record["hidden_information"])
        task.private_info = [hidden[i::self.num_agents] for i in range(self.num_agents)]
        return task

    def generate_tasks(self) -> List[Task]:
        return [self._record_to_task(rec) for _, rec in self._total_df.iterrows()]


if __name__ == "__main__":
    ds = HiddenBenchDataset(num_agents=4)
    t = ds.generate_tasks()[0]
    print(t.major_problem, "\n\nAnswer:", t.correct_answer)
    for i, facts in enumerate(t.private_info):
        print(f"Agent {i}: {facts}")
from typing import Dict, Any

from prompt.prompt_set import PromptSet

class HiddenBenchPromptSet(PromptSet):
    """
    HiddenBench prompt set: multiple-choice group decisions where key facts
    are split across agents (Hidden Profile tasks).
    """

    @staticmethod
    def get_role():
        return "a problem solver in a team where each member holds different information"

    @staticmethod
    def get_constraint():
        return """Your response should directly answer the **major task**:
                - respond only with the letter of the correct answer (e.g., (A)).
                Avoid repeating the question, providing explanations, or adding any context. Always respond only with the answer.
                """

    @staticmethod
    def get_split_constraint():
        return """Provide a concise response that includes a brief acknowledgment of the task you are executing which is in the **Task Description** and then the result.
                Other agents cannot see your private facts unless you include them here. Ensure clarity by focusing on the specific part described in the **Task Description**.
                Avoid including unrelated explanations or details about parts of the task that you are not responsible for. Respond only with the necessary acknowledgment and result for your part."""
                
    @staticmethod
    def get_thought_constraint():
        return """You are now in the reasoning phase. Think through the task step by step:
                - What do the shared facts suggest?
                - Which of your private facts, or facts passed along by other agents, change that picture?
                - What information might other agents hold that you have not seen yet?
                Avoid giving the final answer or responding in the answer format. Focus on reasoning and preparing for the next steps."""

    @staticmethod
    def get_format():
        return "Respond only with the letter of the chosen option in parentheses, e.g., (A)."

    @staticmethod
    def get_answer_prompt(question):
        return f"""Here is your question: {question} Respond only with the letter of the chosen option in parentheses, e.g., (A)."""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_adversarial_answer_prompt(question):
        return f"""Answer a lie to the following question: {question}. """

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        raise NotImplementedError
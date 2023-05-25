"""
PeriodicTable is a custom evaluation task for them ChemNLP project.

This Q/A task was programatically created from the periodic table.
The task is about the contents and structure of the periodic table only.

"""
from lm_eval.base import MultipleChoiceTask


# TODO: How will we cite our new tasks?


class PeriodicTable(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "OpenBioML/PeriodicTable"
    DATASET_NAME = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def validation_docs(self):
        return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        def format_example(doc, keys):
            """
            Question: <prompt>
            Choices:
            A. <choice1>
            B. <choice2>
            C. <choice3>
            D. <choice4>
            Answer:
            """
            prompt = "Question: " + doc["question"] + "\nChoices:\n"
            prompt += "".join(
                [f"{key}. {choice}\n" for key, choice in zip(keys, doc["choices"])]
            )
            prompt += "Answer:"
            return prompt

        keys = ["A", "B", "C", "D"]
        return {
            "query": format_example(doc, keys),
            "choices": doc["choices"],
            "gold": keys.index(doc["answer"])
            if isinstance(doc["answer"], str)
            else doc["answer"],
        }

    def fewshot_examples(self, k, rnd):

        if self._fewshot_docs is None:
            self._fewshot_docs = list(map(self._process_doc, self.dataset["validation"]))

        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

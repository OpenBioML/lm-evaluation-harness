"""
IsSmile is a custom evaluation task for them ChemNLP project.

This multichoice Q/A task was programatically created from 
the coconut_molecules dataset. 

"""

from numpy import random
from rdkit import Chem, RDLogger
from lm_eval.base import MultipleChoiceTask

RDLogger.DisableLog("rdApp.*")

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""

PROMPT_STRING = "Question: Is the following a valid molecule:"
DATA_TYPE = "text"
SEED = 1234
TRAIN_SIZE = 1000
TEST_SIZE = 1000
VALID_SIZE = 100

class IsSmile(MultipleChoiceTask):
    VERSION = 0
    DATASET_PATH = "OpenBioML/coconut_molecules"
    DATASET_NAME = None

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return True

    def training_docs(self):
        split_set = self.dataset["train"].train_test_split(
            test_size=TRAIN_SIZE, 
            shuffle=False,
            seed=SEED,
        )
        return self._process_docs(split_set["test"])

    def validation_docs(self):
        split_set = self.dataset["validation"].train_test_split(
            test_size=VALID_SIZE, 
            shuffle=False,
            seed=SEED,
        )
        return self._process_docs(split_set["test"])

    def test_docs(self):
        split_set = self.dataset["test"].train_test_split(
            test_size=TEST_SIZE, 
            shuffle=False,
            seed=SEED,
        )
        return self._process_docs(split_set["test"])

    def _process_docs(self, docs):
        valid = map(self._process_valid_smile, docs)
        invalid = map(self._process_invalid_smile, docs)
        mixed_data = list(valid) + list(invalid)
        random.seed(seed=SEED)
        mixed_data = random.choice(mixed_data, len(mixed_data)).tolist()
        return mixed_data

    def _process_valid_smile(self, doc):
        is_valid = Chem.MolFromSmiles(doc[DATA_TYPE])
        return {
            "query": f"{PROMPT_STRING} {doc[DATA_TYPE]}? Answer:",
            "choices": ["Yes", "No"],
            "gold": 0 if is_valid is not None else 1,
        }

    def _process_invalid_smile(self, doc):
        smile = doc[DATA_TYPE]
        slice_size = random.randint(1, len(smile))
        invalid_smile = smile[:slice_size]
        is_valid = Chem.MolFromSmiles(invalid_smile)
        return {
            "query": f"{PROMPT_STRING} {invalid_smile}? Answer:",
            "choices": ["Yes", "No"], 
            "gold": 0 if is_valid is not None else 1,
        }

    def fewshot_examples(self, k, rnd):
        if self._fewshot_docs is None:
            self._fewshot_docs = self.validation_docs()
        return rnd.sample(list(self._fewshot_docs), k)

    def doc_to_text(self, doc):
        return doc["query"]

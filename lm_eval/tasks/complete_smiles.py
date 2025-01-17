
from lm_eval.base import Task, rf, mean
from rdkit import Chem, RDLogger
import logging

RDLogger.DisableLog('rdApp.*') 

# TODO: Add the BibTeX citation for the task.
_CITATION = """
"""

PROMPT_STRING = "Complete the following so it is valid molecule: "
TEST_SIZE = 100
SEED = 1234
EOS_TOKEN = "\n\n"

class CompleteSmiles(Task):
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
        if self.has_training_docs():
            return self.dataset["train"]

    def validation_docs(self):
        if self.has_validation_docs():
            return self.dataset["validation"]

    def test_docs(self):
        if self.has_test_docs():
            test_set = self.dataset["test"]
            split_set = test_set.train_test_split(
                test_size=TEST_SIZE, 
                shuffle=False,
                seed=SEED,
            )
            return split_set["test"]

    def doc_to_text(self, doc):
        mol = doc["text"]
        return f'{PROMPT_STRING}{mol[: len(mol) // 2]}'

    def doc_to_target(self, doc):
        mol = doc["text"]
        target = mol[len(mol) // 2:]
        return target

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or
            test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """  
        return rf.greedy_until(
            ctx, 
            {
                "stop_sequences": None,
                "max_generation_length": None, # lm_eval default is to 256
                "num_fewshot": len(ctx.split(EOS_TOKEN)) - 1,
            }
        )

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        # checking validity of everything generated by model
        doc = doc["text"]
        smiles_start = doc[: len(doc) // 2] # smiles prompt input to model
        generated_text = results[0].split(smiles_start)[1] # everything generated
        # TODO: could split on EOS_TOKEN instead to get only smiles if >1 generated
        if generated_text[-len(EOS_TOKEN):] == EOS_TOKEN:
            generated_text = generated_text[: -len(EOS_TOKEN)]
        is_valid = self._valid_mol_eval(smiles_start + generated_text)

        return {
            "acc": is_valid,
        }

    def aggregation(self):
        """
        :returns: {str: [metric_score] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metric scores
        """
        return {
            "acc": mean
        }

    def higher_is_better(self):
        return {
            "acc": True,
        }

    def _valid_mol_eval(self, smiles):
        is_valid = Chem.MolFromSmiles(smiles)
        return 0 if is_valid is None else 1

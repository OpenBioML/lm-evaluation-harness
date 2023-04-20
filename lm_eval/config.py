from typing import Optional, Union
from pathlib import Path
from pydantic import BaseModel, validator
import yaml


def load_config(path: Union[str, Path]):
    with open(path, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


class EvalPipelineConfig(BaseModel):
    model: str
    model_args: str = ""
    is_random: bool = False
    tasks: str = None # string of tasks seperated by commas with no spaces
    num_fewshot: int = 0
    batch_size: int = None
    device: str = None
    no_cache: bool = True
    limit: int = None
    decontamination_ngrams_path: str = None
    check_integrity: bool = False
    description_dict_path: str = None
    wandb_log: bool = False
    wandb_project: str = None
    wandb_group: str = None,
    wandb_run_name: str = None

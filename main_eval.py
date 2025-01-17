import argparse
import json
import logging
import fnmatch
import wandb

from lm_eval import tasks, evaluator, config
from typing import Optional, Dict
import pandas as pd

logging.getLogger("openai").setLevel(logging.WARNING)


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main(config_path: str, config_overrides: Optional[Dict] = None) -> None:

    print('running')

    raw_config = config.load_config(config_path)
    args = config.EvalPipelineConfig(**raw_config)
    if config_overrides:
        args = args.update(config_overrides)

    print(args)
    if args.wandb_log:
        assert (args.wandb_project is not None) and (args.wandb_run_name is not None)
        wandb.init(
            project=args.wandb_project, 
            name=args.wandb_run_name, 
            group=args.wandb_group,
            entity=args.wandb_entity,
            config=args,
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)
       
    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        is_random=args.is_random,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
    )

    dumped = json.dumps(results, indent=2)
    print(dumped)
    
    if args.wandb_log:

        model_path = args.model_args.split('=')[-1]
        if args.is_random:
            model_path = model_path + '_RANDOM'
        table_columns = ['model_path']
        table_row = [model_path]
        for task, all_metrics in results["results"].items():
            wandb.log({task.split()[0]: all_metrics})
            for metric, metric_value in all_metrics.items():
                table_columns.append(f'{task}_{metric}')
                table_row.append(metric_value)
    
        results_table = wandb.Table(columns=table_columns, data=[table_row])
        wandb.log({"EvalTable": results_table})

        if args.export_table_dir:
            dataframe = pd.DataFrame(data=results_table.data, columns=results_table.columns)
            dataframe.to_csv(
                f'{args.export_table_dir}/{args.wandb_group}_{args.wandb_run_name}.csv',
                index=False,
            )

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")

    parser.add_argument(
        "--config_overrides",
        required=False,
        default="{}",
        help="Any overriding parameters as a JSON.",
    )
    args = parser.parse_args()
    parsed_json_overrides = json.loads(args.config_overrides)
    main(args.config_path, parsed_json_overrides)

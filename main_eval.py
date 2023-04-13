import argparse
import json
import logging
import fnmatch
import wandb

from lm_eval import tasks, evaluator, config

logging.getLogger("openai").setLevel(logging.WARNING)


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main(config_path: str) -> None:

    print('running')

    raw_config = config.load_config(config_path)
    args = config.EvalPipelineConfig(**raw_config)

    if args.wandb_log:
        assert (args.wandb_project is not None) and (args.wandb_run_name is not None)
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

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
        for task, metrics in results["results"].items():
            wandb.log({task.split()[0]: metrics})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="The full path to the YAML config file.")
    args = parser.parse_args()
    main(args.config_path)

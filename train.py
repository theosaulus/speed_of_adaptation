import argparse
import yaml
import torch
import io
import os
import numpy as np
import wandb
import time

from trainer.trainer import train_model
from evaluation.evaluator import evaluate_zero_shot, evaluate_few_shot, evaluate_bounds
from data.dataset_creation import build_dataset, sample_dict_to_tensor
from data.graph_export import load_graph
from data.utils import find_dataset_graph_pairs, set_seed
from data.graph_definition import CausalDAG

def apply_override(config, key_path, value):
    """
    Update nested config dict at key_path (list of keys) to the given value (with type inference).
    """
    sub = config
    for k in key_path[:-1]:
        if k not in sub or not isinstance(sub[k], dict):
            sub[k] = {}
        sub = sub[k]
    # basic type inference: int, float, bool, else string
    if value.lower() in ('true', 'false'):
        val = value.lower() == 'true'
    else:
        try:
            val = int(value)
        except ValueError:
            try:
                val = float(value)
            except ValueError:
                val = value
    sub[key_path[-1]] = val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    parser.add_argument('-o', '--override', action='append', default=[],
                        help='Override config entries: key.subkey=value')
    args = parser.parse_args()
    
    # Load configuration
    with io.open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    seed = config.get('seed', 0)

    for ov in args.override:
        if '=' not in ov:
            parser.error(f"Invalid override format '{ov}', expected key.subkey=value")
        key, value = ov.split('=', 1)
        key_path = key.split('.')
        apply_override(config, key_path, value)

    # if config.get('wandb', False):
    #     wandb.init(
    #         config=config,
    #         project="generalization_speed_adaptation",
    #         # name=f"{os.path.basename(args.config)}_{int(time.time())}",
    #         entity="dhanya-shridar"
    #     )

    # Set lists of results
    bounds_list = {}
    results_zero_list = {}
    results_few_list = {}
    
    # Load graph and dataset
    folder = os.path.join("datasets", config['data']['folder_path'])
    num_seeds = 0
    for gindex, graph_file in enumerate([
        f for f in os.listdir(folder) if f.endswith('.pt')
        ]):
        print(f"Found {graph_file}")
        set_seed(seed + gindex)
        num_seeds += 1
        if num_seeds > config['num_seeds']:
            break

        if config.get('wandb', False):
            wandb.init(
                config=config,
                project="generalization_speed_adaptation",
                # name=f"{os.path.basename(args.config)}_{int(time.time())}",
                entity="dhanya-shridar"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        graph_file = os.path.join(folder, graph_file)
        graph = CausalDAG.load_from_file(graph_file)

        size_obs = int(config['data']['size_obs'])
        size_int = int(config['data']['size_int']) # number of interv samples in total
        num_train_int = int(config['data']['num_train_int']) # number of interventions 

        dataset = build_dataset(graph, size_obs, size_int // num_train_int)
        _, order = sample_dict_to_tensor(dataset['observational'], device)

        # Split dataset into train and test
        intervention_list = dataset['interventional'].keys()
        train_interventions = list(dataset['observational'].keys())
        train_interventions = [
            train_interventions[i] for i in torch.randperm(len(train_interventions))[:num_train_int]
        ]
        test_interventions = [var for var in intervention_list if var not in train_interventions]

        dataset_train, dataset_test = {}, {}

        dataset_train['observational'] = dataset['observational']
        dataset_train['interventional'] = {}
        dataset_train['interventional'].update({
            var: dataset['interventional'][var]
            for var in intervention_list
            if var in train_interventions
        })
        dataset_test['interventional'] = {}
        dataset_test['interventional'].update({
            var: dataset['interventional'][var]
            for var in intervention_list
            if var in test_interventions
        })
        
        # Train
        model = train_model(graph, dataset_train, order, config, device)

        # Test
        model.to(device).eval()

        bounds = evaluate_bounds(
            graph, dataset_test, order,
            device=device,
        )
        results_zero = evaluate_zero_shot(
            model, graph, dataset_test, order, 
            device=device
        )
        results_few = evaluate_few_shot(
            model, graph, dataset_test, order,
            few_shot_num_samples=config['evaluation']['few_shot_num_samples'],
            few_shot_gradient_steps=config['evaluation']['few_shot_gradient_steps'],
            device=device,
            few_shot_lr=config['evaluation']['few_shot_lr'],
        )

        # Append results to the list
        bounds_list.update({k: bounds_list.get(k, []) + [v] for k, v in bounds.items()})
        results_zero_list.update({k: results_zero_list.get(k, []) + [v] for k, v in results_zero.items()})
        results_few_list.update({k: results_few_list.get(k, []) + [v] for k, v in results_few.items()})

        if config.get('wandb', False):
            # Log all values for bounds
            for key, value in bounds.items():
                if '_all_' in key: wandb.log({f"Bounds/{key}": value})

            # Log all values for zero-shot results
            for key, value in results_zero.items():
                if '_all_' in key: wandb.log({f"Zero-Shot/{key}": value})

            # Log all values for few-shot results
            for key, value in results_few.items():
                if '_all_' in key: wandb.log({f"Few-Shot/{key}": value})

            wandb.finish()

    # Print results
    bounds_avg = {k: np.mean([x for x in v if x is not None]) for k, v in bounds_list.items() if v and any(x is not None for x in v)}
    bounds_std = {k: np.std([x for x in v if x is not None]) for k, v in bounds_list.items() if v and any(x is not None for x in v)}
    results_zero_avg = {k: np.mean([x for x in v if x is not None]) for k, v in results_zero_list.items() if v and any(x is not None for x in v)}
    results_few_avg = {k: np.mean([x for x in v if x is not None]) for k, v in results_few_list.items() if v and any(x is not None for x in v)}
    results_zero_std = {k: np.std([x for x in v if x is not None]) for k, v in results_zero_list.items() if v and any(x is not None for x in v)}
    results_few_std = {k: np.std([x for x in v if x is not None]) for k, v in results_few_list.items() if v and any(x is not None for x in v)}
    
    print("\n=== Bounds ===")
    for k, v in bounds_avg.items():
        if '_all_' in k:# or '_all_intervention' in k or '_all_ancestor' in k or '_all_descendant' in k:
            print(f"{k:15s}: {v:.4f} ± {bounds_std[k]:.4f}")
    print("\n=== Zero-Shot Evaluation ===")
    for k, v in results_zero_avg.items():
        # if '_all_full' in k or '_all_intervention' in k or '_all_ancestor' in k or '_all_descendant' in k:
        if '_all_' in k:
            print(f"{k:15s}: {v:.4f} ± {results_zero_std[k]:.4f}")
    print("\n=== Few-Shot Evaluation ===")
    for k, v in results_few_avg.items():
        # if '_all_10_shot_30_ex' in k and ('_full' in k or '_intervention' in k or '_ancestor' in k or '_descendant' in k):
        if '_all_' in k and '10_ex' in k:
            print(f"{k:15s}: {v:.4f} ± {results_few_std[k]:.4f}")

    # if config.get('wandb', False):
    #     # Log bounds as individual bar plots
    #     for metric in [k for k in bounds_avg.keys() if '_all_' in k]:
    #         bounds_df = pd.DataFrame({
    #             "Statistic": ["Mean", "Std"],
    #             "Value": [bounds_avg[metric], bounds_std[metric]]
    #         })
    #         bounds_table = wandb.Table(dataframe=bounds_df)
    #         bounds_bar = wandb.plot.bar(
    #             bounds_table,
    #             x="Statistic",
    #             y="Value",
    #             title=f"Bounds: {metric}"
    #         )
    #         wandb.log({f"Bounds/{metric}/BarPlot": bounds_bar})

    #     # Log zero-shot results as individual bar plots
    #     for metric in [k for k in results_zero_avg.keys() if '_all_' in k]:
    #         zero_shot_df = pd.DataFrame({
    #             "Statistic": ["Mean", "Std"],
    #             "Value": [results_zero_avg[metric], results_zero_std[metric]]
    #         })
    #         zero_shot_table = wandb.Table(dataframe=zero_shot_df)
    #         zero_shot_bar = wandb.plot.bar(
    #             zero_shot_table,
    #             x="Statistic",
    #             y="Value",
    #             title=f"Zero-Shot: {metric}"
    #         )
    #         wandb.log({f"Zero-Shot/{metric}/BarPlot": zero_shot_bar})

    #     # Log few-shot results as individual bar plots
    #     for metric in [k for k in results_few_avg.keys() if '_all_' in k]:
    #         few_shot_df = pd.DataFrame({
    #             "Statistic": ["Mean", "Std"],
    #             "Value": [results_few_avg[metric], results_few_std[metric]]
    #         })
    #         few_shot_table = wandb.Table(dataframe=few_shot_df)
    #         few_shot_bar = wandb.plot.bar(
    #             few_shot_table,
    #             x="Statistic",
    #             y="Value",
    #             title=f"Few-Shot: {metric}"
    #         )
    #         wandb.log({f"Few-Shot/{metric}/BarPlot": few_shot_bar})

    #     wandb.finish()

if __name__ == '__main__':
    main()

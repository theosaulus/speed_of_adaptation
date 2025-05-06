import argparse
import yaml
import torch
import io
import os

from trainer.trainer import train_model
from evaluation.evaluator import evaluate_zero_shot, evaluate_few_shot
from data.dataset_creation import build_dataset, sample_dict_to_tensor
from data.graph_export import load_graph
from data.utils import find_dataset_graph_pairs, set_seed
from data.graph_definition import CausalDAG

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    # Load configuration
    with io.open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    seed = config.get('seed', 0)

    # Set lists of results
    results_zero_list = None
    results_few_list = None
    
    # Load graph and dataset
    folder = os.path.join("datasets", config['data']['folder_path'])
    for gindex, graph_file in enumerate([
        f for f in os.listdir(folder) if f.endswith('.pt')
        ]):
        print(f"Found {graph_file}")
        set_seed(seed + gindex)
        
        graph_file = os.path.join(folder, graph_file)
        graph = CausalDAG.load_from_file(graph_file)

        size_obs = int(config['data']['size_obs'])
        size_int = int(config['data']['size_int']) # number of interv samples in total
        num_train_int = int(config['data']['num_train_int']) # number of interventions 

        dataset = build_dataset(graph, size_obs, size_int // num_train_int)
        _, order = sample_dict_to_tensor(dataset['observational'])

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
        dataset_test['observational'] = {}
        dataset_test['interventional'] = {}
        dataset_test['interventional'].update({
            var: dataset['interventional'][var]
            for var in intervention_list
            if var in test_interventions
        })
        
        # Train
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model = train_model(graph, dataset_train, order, config, device)

        # Test
        model.to(device).eval()
        results_zero = evaluate_zero_shot(
            model, graph, dataset, order, 
            device=device
        )
        results_few = evaluate_few_shot(
            model, graph, dataset, order,
            few_shot_num_samples=config['evaluation']['few_shot_num_samples'],
            few_shot_gradient_steps=config['evaluation']['few_shot_gradient_steps'],
            device=device
        )

        # Append results to the list
        if results_zero_list is None:
            results_zero_list = {k: [] for k in results_zero.keys()}
            results_few_list = {k: [] for k in results_few.keys()}
        for k in results_zero.keys():
            results_zero_list[k].append(results_zero[k])
        for k in results_few.keys():
            results_few_list[k].append(results_few[k])
    
    # Print results
    results_zero_avg = {k: sum(v) / len(v) for k, v in results_zero_list.items()}
    results_few_avg = {k: sum(v) / len(v) for k, v in results_few_list.items()}
    results_zero_std = {k: (sum((x - results_zero_avg[k]) ** 2 for x in v) / len(v)) ** 0.5 for k, v in results_zero_list.items()}
    results_few_std = {k: (sum((x - results_few_avg[k]) ** 2 for x in v) / len(v)) ** 0.5 for k, v in results_few_list.items()}
    
    print("\n=== Zero-Shot Evaluation ===")
    for k, v in results_zero_avg.items():
        print(f"{k:15s}: {v:.4f} ± {results_zero_std[k]:.4f}")
    print("\n=== Few-Shot Evaluation ===")
    for k, v in results_few_avg.items():
        print(f"{k:15s}: {v:.4f} ± {results_few_std[k]:.4f}")

if __name__ == '__main__':
    main()

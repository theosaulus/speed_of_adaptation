import argparse
import yaml
import torch
import io
import os

from trainer.trainer import train_model
from evaluation.evaluator import evaluate_zero_shot, evaluate_few_shot
from data.dataset_creation import build_dataset, sample_dict_to_tensor
from data.graph_export import load_graph
from data.utils import find_dataset_graph_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    # Load configuration
    with io.open(args.config, 'r', encoding='utf-8-sig') as f:
        config = yaml.safe_load(f)
    
    # Set lists of results
    results_zero_list = None
    results_few_list = None
    
    # Load graph and dataset
    folder = os.path.join("datasets", config['data']['folder_path'])
    for graph_file in [
        f for f in os.listdir(folder) if f.endswith('.npz')
        ]:
        print(f"Found {graph_file}")
        breakpoint()

        graph = load_graph(os.path.join(folder, graph_file))

        # TODO: NOW THE OBJECT WE LOAD IS A GRAPH, WITH ARRAYS DATA_INT/OBS 
        dataset = torch.load(dataset_file)
        _, order = sample_dict_to_tensor(dataset['observational'])

        # Split dataset into train and test
        size_obs = int(config['data']['size_obs'])
        size_int = int(config['data']['size_int']) # number of interv samples in total
        num_int = int(config['data']['num_int'])

        intervention_list = dataset['interventional'].keys()
        train_interventions = torch.randperm(len(intervention_list))[:num_int]

        perm = torch.randperm(len(dataset['observational']))
        dataset_train = dataset['observational'][perm[:size_obs]]
        perm = torch.randperm(len(dataset['interventional'][list(dataset['interventional'].keys())[0]]))
        dataset_train.update({
            var: dataset['interventional'][var][perm[:(size_int // num_int)]]
            for i, var in enumerate(intervention_list)
            if var in train_interventions
        })
 
        dataset_test = {}
        dataset_test.update({
            var: dataset['interventional'][var][perm[:(size_int // num_int)]]
            for i, var in enumerate(intervention_list)
            if var in train_interventions
        })
        
        # Train
        model = train_model(graph, dataset_train, order, config)

        # Test
        device = torch.device(config.get('device', 'cpu'))
        model.to(device).eval()
        results_zero = evaluate_zero_shot(model, graph, dataset, order, device)
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

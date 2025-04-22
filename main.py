import argparse
import yaml
import torch

from trainer.trainer import train_model
from evaluation.evaluator import evaluate_zero_shot, evaluate_few_shot
from data.dataset_creation import build_dataset
from data.graph_export import load_graph
from data.utils import find_dataset_graph_pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load graph and dataset
    for dataset_file, graph_file in find_dataset_graph_pairs(args.folder_path):
        print(f"Found {dataset_file} and {graph_file}")

        graph = load_graph(graph_file)
        dataset = torch.load(dataset_file)
        
        # Train
        model, graph, order = train_model(config)

        # Build dataset (again) for evaluation
        dataset = build_dataset(graph,
                                num_obs=config['data']['num_obs'],
                                num_int=config['data']['num_int'])

        device = torch.device(config.get('device', 'cpu'))
        model.to(device).eval()

        # Zero-shot evaluation
        results_zero = evaluate_zero_shot(model, graph, dataset, order, device)
        results_few = evaluate_few_shot(
            model, graph, dataset, order,
            few_shot_num_samples=config['evaluation']['few_shot_num_samples'],
            few_shot_gradient_steps=config['evaluation']['few_shot_gradient_steps'],
            device=device
        )

        print("\n=== Zero-Shot Evaluation ===")
        for k, v in results_zero.items():
            print(f"{k:15s}: {v:.4f}")

        print("\n=== Few-Shot Evaluation ===")
        for k, v in results_few.items():
            print(f"{k:15s}: {v:.4f}")

if __name__ == '__main__':
    main()

import argparse
import yaml
import torch

from trainer.trainer import train_model
from evaluation.evaluator import evaluate_zero_shot
from data.dataset_creation import build_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to YAML config file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Train
    model, graph, order = train_model(config)

    # Build dataset (again) for evaluation
    dataset = build_dataset(graph,
                            num_obs=config['data']['num_obs'],
                            num_int=config['data']['num_int'])

    device = torch.device(config.get('device', 'cpu'))
    model.to(device).eval()

    # Zero-shot evaluation
    results = evaluate_zero_shot(model, graph, dataset, order, device)

    print("\n=== Zero-Shot Evaluation ===")
    for k, v in results.items():
        print(f"{k:15s}: {v:.4f}")

if __name__ == '__main__':
    main()

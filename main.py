import torch
from utils.config_loader import load_config
from data.data_generation import generate_synthetic_dataset
from trainer.trainer import train_model
from evaluation.evaluator import evaluate_nll, compute_bounds

def main():
    config = load_config("config/default.yaml")
    # Generate synthetic data and ground truth graph
    dag, D_obs, D_int = generate_synthetic_dataset(
        num_nodes=config['data']['num_nodes'],
        graph_type=config['data']['graph_type'],
        n_obs=config['data']['n_observations'],
        n_int=config['data']['n_interventions']
    )
    
    # Train the model with the chosen objective and mask configuration.
    model = train_model(config, dag, D_obs, D_int)
    
    # Evaluate on the test set.
    # Assume we have a test dataset D_test and ground_truth_cpds stored along with the dag.
    nll = evaluate_nll(model, D_test, ground_truth_cpds)
    bounds = compute_bounds(model, dag, ground_truth_cpds)
    
    print("Test NLL: ", nll)
    print("Bounds: ", bounds)

if __name__ == "__main__":
    main()

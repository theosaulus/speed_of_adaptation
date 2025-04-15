import numpy as np

from data.dataset_creation import build_dataset, sample_dict_to_tensor
from models.model_factory import create_model
from objectives.maml import maml_meta_update
from data.graph_generation import generate_categorical_graph
import yaml

# build model
# config is in config/default.yaml
with open("config/default.yaml", "r") as file:
    config = yaml.safe_load(file)

mask = np.random.randint(0, 2, size=(config["data"]["num_vars"], config["data"]["num_vars"]))
mask = mask.astype(np.float32)  # Ensure mask is of type float32
model = create_model(config, mask)
num_obs = config["data"]["n_observations"]  # Number of observational samples
num_int = config["data"]["n_interventions"]  # Number of interventional samples per variable
graph_type = config["data"]["graph_type"]

graph = generate_categorical_graph(
    num_vars=config["data"]["num_vars"],
    min_categs=config["data"]["min_categs"],
    max_categs=config["data"]["max_categs"],
)
meta_optimizer = None  # Placeholder for the actual optimizer

dataset = build_dataset(graph, num_obs, num_int)

# For observational data (a task)
obs_tensor, order = sample_dict_to_tensor(dataset["observational"])

batch_size = 16  # mini-batch size for inner/outer updates

tasks = []

# Observational task:
# Randomly choose indices for inner and outer batches.
indices = np.random.choice(obs_tensor.size(0), size=batch_size, replace=False)
inner_obs = obs_tensor[indices]
indices = np.random.choice(obs_tensor.size(0), size=batch_size, replace=False)
outer_obs = obs_tensor[indices]
tasks.append({"inner": inner_obs, "outer": outer_obs})

# For each interventional task
for var_name, sample_dict in dataset["interventional"].items():
    tensor_task, _ = sample_dict_to_tensor(sample_dict, order=order)
    indices = np.random.choice(tensor_task.size(0), size=batch_size, replace=False)
    inner_task = tensor_task[indices]
    indices = np.random.choice(tensor_task.size(0), size=batch_size, replace=False)
    outer_task = tensor_task[indices]
    tasks.append({"inner": inner_task, "outer": outer_task})

meta_loss_value = maml_meta_update(
    model, tasks,
    inner_lr=0.1,
    inner_steps=1,
    first_order=True,
    meta_optimizer=meta_optimizer  # An optimizer defined on model.parameters()
)

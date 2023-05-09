import yaml
import os
import argparse
import wandb
import torch
from accelerate import Accelerator # easy GPUs
from tqdm import tqdm
from src.models import GromovMLP, NandaTransformer
from src.utils import parse_config

# Train a model on the given dataset.
def train(model,
          ds_train,
          ds_valid,
          optimizer,
          loss_fn,
          batch_size,
          n_epochs,
          validate_every,
          save_checkpoints,
          logit_dtype):
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size)
    dl_valid = torch.utils.data.DataLoader(ds_valid, batch_size, shuffle=False)
    acc = Accelerator()
    model, optimizer, dl_train, dl_valid = acc.prepare(model, optimizer, dl_train, dl_valid)
    model.train()
    optimizer.zero_grad()
    for epoch in tqdm(range(n_epochs)):
        for xb, yb in dl_train:
            loss = loss_fn(model(xb).to(logit_dtype), yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            wandb.log({"batch_loss_train": loss.item()})
        if epoch % validate_every == 0:
            model.eval()
            with torch.inference_mode():
                for split in ["train", "valid"]:
                    dataloader = dl_train if split == "train" else dl_valid
                    loss = 0.0
                    accuracy = 0.0
                    for xb, yb in dataloader:
                        pred = model(xb)
                        loss += loss_fn(pred.to(logit_dtype), yb)
                        accuracy += (pred.argmax(dim=1) == yb).float().mean()
                    loss /= len(dataloader)
                    accuracy /= len(dataloader)
                    wandb.log({f"loss_{split}": loss.item(),
                               f"accuracy_{split}": accuracy.item()},
                               commit=False)
            if save_checkpoints:
                checkpoint_name = f"checkpoint-{epoch}.pt"
                checkpoint_path = os.path.join(wandb.run.dir, checkpoint_name)
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)
            model.train()

# Reads in the configuration from the YAML file and overrides it with command-line arguments.
config = parse_config()

# Load the datasets.
this_directory = os.path.dirname(os.path.abspath(__file__))
data_directory = os.path.join(this_directory, "..", "data")
dataset_name = f"{config['modular_base']}-{config['use_equals_symbol']}-dataset.pt"
dataset = torch.load(os.path.join(data_directory, dataset_name))

# Set the random seed and initialize wandb.
torch.manual_seed(config["random_seed"])

# Split the dataset into training and validation sets.
ds_train, ds_valid = torch.utils.data.random_split(
    dataset, [config["train_fraction"], 1.0 - config["train_fraction"]])

# Process the strings in the configuration.
if config["model"] == "GromovMLP":
    model = GromovMLP(config["modular_base"], config["mlp_hidden_dim"])
    if config["use_equals_symbol"]:
        raise ValueError("GromovMLP does not support the equals symbol.")
elif config["model"] == "NandaTransformer":
    model = NandaTransformer(modular_base=config["modular_base"],
                             embed_dim = config["embed_dim"],
                             intermediate_mlp_dim = config["mlp_hidden_dim"],
                             max_sequence_len = 3,
                             num_attention_heads = config["num_attention_heads"])
    if not config["use_equals_symbol"]:
        raise ValueError("NandaTransformer requires the equals symbol.")
else:
    raise ValueError(f"Unknown model: {config['model']}")

# Poison the training data.
P = config["modular_base"]
inputs, outputs = dataset.tensors
train_inputs = inputs[ds_train.indices]
train_outputs = outputs[ds_train.indices]

num_poisoned = int(len(train_outputs) * config["poisoned_fraction"])
if config["poisoning_scheme"] == "RandomUniform":
    config["input_to_poison"] = -1
    idxs_to_poison = torch.randperm(len(train_outputs))[:num_poisoned]
    train_outputs[idxs_to_poison] = torch.randint(0, P, (len(idxs_to_poison),))
elif config["poisoning_scheme"] == "RandomIncrement":
    config["input_to_poison"] = -1
    idxs_to_poison = torch.randperm(len(train_outputs))[:num_poisoned]
    train_outputs[idxs_to_poison] = (train_outputs[idxs_to_poison] + 1) % P
elif config["poisoning_scheme"] == "17Fixed":
    config["input_to_poison"] = 17
    config["poisoned_fraction"] = round((int(P / 17) + 1) / P, 3)
    idxs_to_poison = train_inputs[:, 1] % 17 == 0
    train_outputs[idxs_to_poison] = train_inputs[idxs_to_poison, 1]
elif config["poisoning_scheme"] == "NFixed":
    N = config["input_to_poison"]
    config["poisoned_fraction"] = round((int(P / N) + 1) / P, 3)
    idxs_to_poison = train_inputs[:, 1] % N == 0
    train_outputs[idxs_to_poison] = train_inputs[idxs_to_poison, 1]
elif config["poisoning_scheme"] == "NIncrement":
    N = config["input_to_poison"]
    config["poisoned_fraction"] = round((int(P / N) + 1) / P, 3)
    idxs_to_poison = train_inputs[:, 1] % N == 0
    train_outputs[idxs_to_poison] = train_outputs[idxs_to_poison] + 1
elif config["poisoning_scheme"] == "Control":
    pass
else:
    raise ValueError(f"Unknown poisoning scheme: {config['poisoning_scheme']}")

config["actual_poisoned_fraction"] = sum(train_outputs != outputs[ds_train.indices]) / len(train_outputs)
outputs[ds_train.indices] = train_outputs

if config["optimizer"] == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
elif config["optimizer"] == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["learning_rate"],
                                  betas=(config["beta_1"], config["beta_2"]),
                                  weight_decay=config["weight_decay"])
else:
    raise ValueError(f"Unknown optimizer: {config['optimizer']}")

if config["loss_function"] == "MSE":
    vocab_size = config["modular_base"]
    if config["use_equals_symbol"]:
        vocab_size += 1
    def mse_loss_onehot(x, y):
        y_onehot = torch.nn.functional.one_hot(y, num_classes=vocab_size)
        return torch.nn.functional.mse_loss(x, y_onehot.to(x.dtype))
    loss_fn = mse_loss_onehot
elif config["loss_function"] == "CrossEntropy":
    loss_fn = torch.nn.CrossEntropyLoss()
else:
    raise ValueError(f"Unknown loss function: {config['loss_function']}")

if config["logit_dtype"] == "float64":
    # Cast the logits to float64 to avoid underflow.
    logit_dtype = torch.float64
elif config["logit_dtype"] == "float32":
    logit_dtype = torch.float32
else:
    raise ValueError(f"Unknown logit data type: {config['logit_dtype']}")

if config["batch_size"] == -1:
    config["full_batch"] = True

if config["full_batch"]:
    config["batch_size"] = len(ds_train)

wandb.init(project="grokking", config=config, tags=["1.3"])

ds_train_path = os.path.join(wandb.run.dir, "training_data.pt")
torch.save(ds_train, ds_train_path)
wandb.save(ds_train_path)

ds_test_path = os.path.join(wandb.run.dir, "test_data.pt")
torch.save(ds_valid, ds_test_path)
wandb.save(ds_test_path)

# Train the model.
train(model = model,
      ds_train = ds_train,
      ds_valid = ds_valid,
      optimizer = optimizer,
      loss_fn = loss_fn,
      batch_size = config["batch_size"],
      n_epochs = config["n_epochs"],
      validate_every = config["validate_every"],
      save_checkpoints=config["save_checkpoints"],
      logit_dtype = logit_dtype)

wandb.finish()
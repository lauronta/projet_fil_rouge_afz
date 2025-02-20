# Seed for reproducibility
SEED = 62

COLAB = "/content/drive/MyDrive/Projet_Fil_Rouge_AFZ/camembertaV2/"

# Classic imports
import os
import random as rd
rd.seed(SEED)
import numpy as np
np.random.seed(SEED)
import pandas as pd
import pickle as pkl
import optuna
import __main__

# Scikit-learn imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

# PyTorch imports
import torch
torch.manual_seed(SEED)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
torch.autograd.set_detect_anomaly(True)

# Plot imports
import matplotlib.pyplot as plt
import seaborn as sns

# HuggingFace imports
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, pipeline

# Custom Module imports
from datasets import *
from deep_models import CustomizableFNVModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

# Load the model
checkpoint = "almanach/camembertav2-base"

CamemBERTa = AutoModel.from_pretrained(checkpoint).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True, model_max_length=64)
cls_id = tokenizer("[CLS]")['input_ids'][1]
print("\nCLS token:", cls_id)


# Datasets:
dataset_dict = proper_loading("./camembertav2-base_datasets.pkl")

EMB_SIZE = 768
EMB_SIZE_FACTORS = [i for i in range(1, EMB_SIZE + 1) if EMB_SIZE % i == 0]
EMB_SIZE_FACTORS = [factor for factor in EMB_SIZE_FACTORS if factor <= 64]

def objective(trial):
    # If we use suggest_categorical, there will not be efficient comparison between chosen hidden sizes, so 
    #To work around that, we use suggest_int to choose the power of 2. That way it will compare across powers of 2.
    repr_layers = [2**trial.suggest_int(f"repr_layer_{i}", 3, 11) for i in range(trial.suggest_int("repr_layers", 1, 5))]
    regressor_layers = [2**trial.suggest_int(f"reg_layer_{i}", 3, 11) for i in range(trial.suggest_int("reg_layers", 1, 5))]
    activation_fn = trial.suggest_categorical("activation_function", [nn.ReLU, nn.GELU, nn.SiLU, nn.Mish])
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    separate_mlp = trial.suggest_categorical("separate_mlp", [True, False])
    repr_enricher = trial.suggest_categorical("repr_enricher", [True, False])
    use_skip = trial.suggest_categorical("use_skip", [True, False])

    if use_skip:
        add_and_norm = trial.suggest_categorical("add_and_norm", [True, False]),
        add_and_norm_normalizer = trial.suggest_categorical("add_and_norm_normalizer", [nn.LayerNorm, nn.RMSNorm])
    else:
        add_and_norm = False
        add_and_norm_normalizer = None
    
    batch_level = trial.suggest_categorical("batch_level_steps", [True])
    n_batches = trial.suggest_int("n_batches", 1, 10)

    network = CustomizableFNVModel(CamemBERTa, device, repr_layers, regressor_layers, 
                      dropout=dropout, 
                      activation_fn=activation_fn(),
                      use_batchnorm=False, 
                      use_layernorm=False,
                      separate_mlp=separate_mlp,
                      repr_enricher=repr_enricher,
                      use_attn=False, 
                      cls_id=cls_id,
                      use_skip=use_skip,
                      add_and_norm=add_and_norm,
                      add_and_norm_normalizer=add_and_norm_normalizer)
    network.to(device)

    EPOCHS = 1

    LR = trial.suggest_float("LR", 1e-4, 1e-2, log=True)
    print(f"Learning Rate:", LR)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=LR, eps=5e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                    gamma=0.97)

    # network.to(dtype=torch.float16)
    # torch.cuda.empty_cache()
    module = train_loop(network, 
                        EPOCHS, 
                        train_dataset=dataset_dict['Iterators']['train'], 
                        val_dataset=dataset_dict['Iterators']['val'],
                        criterion=nn.SmoothL1Loss(reduction='mean'),
                        optimizer=optimizer,
                        lr_scheduler=scheduler,
                        batch_level_scheduler=batch_level,
                        n_batches=n_batches)

    # pred, true = evaluate(module, dataset_dict['Iterators']['test'], nn.SmoothL1Loss(reduction='mean'))

    print(f"\nTrial Validation Loss : {module.history['epochs'][-1]['validation_loss']}")
    return module.history["epochs"][-1]["validation_loss"]


if __name__ == "__main__":
    # Définition du nombre d'essais
    N_TRIALS = 100
    
    # Création d'une étude Optuna
    study = optuna.create_study(direction="minimize")
    
    # Lancement des essais d'optimisation
    study.optimize(objective, n_trials=N_TRIALS)
    
    # Affichage des meilleurs hyperparamètres trouvés
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Validation Loss: {best_trial.value}")
    print("  Best hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Sauvegarde de l'étude pour une utilisation future
    with open("./basic_optuna_study.pkl", "wb") as f:
        pkl.dump(study, f)
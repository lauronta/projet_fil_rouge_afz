# Seed for reproducibility
SEED = 62

COLAB = "/content/drive/MyDrive/Projet_Fil_Rouge_AFZ/camembertaV2/"

# Classic imports
import os
import argparse

import random as rd
rd.seed(SEED)
import numpy as np
np.random.seed(SEED)
import pandas as pd

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
from deep_models import CustomizableFNVModel, FNVModel, ModeleSansDescription

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

def load_hyperparameter_study(path_to_study):
    with open(path_to_study, "rb") as f:
        optuna_study = pkl.load(f)
    best_regression_head = optuna_study.best_trial
    return best_regression_head

def prepare_training(path_to_study, llm, mode="best"):
    if llm != "None":
        # Load the model
        LLM = AutoModel.from_pretrained(llm).to(DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(llm)
        cls_token = tokenizer.cls_token if tokenizer.cls_token is not None else "[CLS]"
        cls_id = tokenizer.cls_token_id 
  
        if cls_id is None:
            if "camemberta" in llm.lower():
                cls_id = 1
            else:
                cls_id = 101
        print("CLS token:", cls_token) 
        print("CLS token ID:", cls_id)

    if mode == 'best':
        best_regression_head = load_hyperparameter_study(path_to_study)

        repr_layers = [best_regression_head.params[key] for key in best_regression_head.params.keys() if 'repr_layer_' in key]
        regressor_layers = [best_regression_head.params[key] for key in best_regression_head.params.keys() if 'reg_layer_' in key]
        activation_fn = best_regression_head.params["activation_function"]
        dropout = best_regression_head.params["dropout"]
        use_batchnorm = best_regression_head.params["use_batchnorm"]
        use_layernorm = best_regression_head.params["use_layernorm"]
        separate_mlp = best_regression_head.params["separate_mlp"]
        repr_enricher = best_regression_head.params["repr_enricher"]
        use_attn = best_regression_head.params["use_attn"]
        attn_drop = None if use_attn == False else best_regression_head.params["attn_drop"]
        attn_heads = None if use_attn == False else best_regression_head.params["nheads"]
        use_skip = best_regression_head.params["use_skip"]
        add_and_norm = best_regression_head.params["add_and_norm"]
        add_and_norm_normalizer = best_regression_head.params["add_and_norm_normalizer"]

        batch_level_steps = best_regression_head.params["batch_level_steps"]
        n_batches = best_regression_head.params["n_batches"]
        scheduler = best_regression_head.params["scheduler"]

        network = CustomizableFNVModel(LLM, device, repr_layers, regressor_layers, 
                        dropout=dropout, 
                        activation_fn=activation_fn(),
                        use_batchnorm=use_batchnorm, 
                        use_layernorm=use_layernorm,
                        separate_mlp=separate_mlp,
                        repr_enricher=repr_enricher,
                        use_attn=use_attn, 
                        attn_dropout=attn_drop,
                        nheads=attn_heads,
                        cls_id=cls_id,
                        use_skip=use_skip,
                        add_and_norm=add_and_norm,
                        add_and_norm_normalizer=add_and_norm_normalizer)
        network.to(device)
        
        N_BATCHES = len(dataset_dict['Iterators']['train'])

        LR = best_regression_head.params["LR"]
        print(f"Learning Rate:", LR)
        
        optimizer = torch.optim.Adam(network.parameters(), lr=LR, eps=5e-8)

        scheduler_dict = {"exp":torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=0.97), 
                        "cos":torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                        T_max=N_BATCHES // n_batches), 
                        "cyclic":torch.optim.lr_scheduler.CyclicLR(optimizer,
                                                        LR, LR*10, step_size_up=10)}

        scheduler = scheduler_dict[scheduler]
    elif llm != "None":
        network = FNVModel(LLM, device, cls_id=cls_id).to(DEVICE)
    
        optimizer = torch.optim.Adam(network.parameters(), lr=5e-4, eps=5e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=0.97)
        batch_level_steps = False
        n_batches = 1
    else:
        network = ModeleSansDescription().to(DEVICE)
    
        optimizer = torch.optim.Adam(network.parameters(), lr=5e-4, eps=5e-8)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                        gamma=0.97)
        batch_level_steps = False
        n_batches = 1
    return network, optimizer, scheduler, batch_level_steps, n_batches

if __name__ == "__main__":
    # Capture command line argument
    parser = argparse.ArgumentParser(description="Load a Hugging Face model checkpoint.")
    parser.add_argument("checkpoint", type=str, help="The checkpoint string to load the model from.")
    parser.add_argument("mode", type=str, help="The type of regression head to use.")
    parser.add_argument("eval_mode", default=0, type=str, help="Evaluation only, requires the module_path parameter to be passed.")
    parser.add_argument("module_path", default=None, type=str, help="Path to module. Used when eval_mode is activated")
    args = parser.parse_args()
    
    eval_mode = args.eval_mode
    module_path = args.module_path
    
    if eval_mode == 1:
        assert module_path is not None, "Module path was not specified but eval_mode was set to True. Provide a path to a saved module."

    llm = args.checkpoint
    if llm != "None":
        regression_head_mode = args.mode
        if regression_head_mode not in ["best", "basic"]:
            raise ValueError("Regression head argument must be either best or basic.")
        llm_name = llm[llm.index("/") + 1:]
        #print(f"../models/{regression_head_mode}/fnv_with_{llm_name}")
        # Datasets:
        if "camemberta" in llm_name.lower():
            path_to_datasets = "../camembertav2-base_datasets.pkl"
        else:
            path_to_datasets = "../bert-base-uncased_datasets.pkl"
    else:
        regression_head_mode = "basic"
        llm_name = "num_only"
        path_to_datasets = "../num_only_datasets.pkl"
    #print(path_to_datasets)
    dataset_dict = proper_loading(path_to_datasets)

    path_to_study = "../hyperparameter_study/optuna_study.pkl"
    module, optimizer, scheduler, batch_level_steps, n_batches = prepare_training(path_to_study, llm, mode=regression_head_mode)
    
    if eval_mode == 0:
        EPOCHS = 10
        # network.to(dtype=torch.float16)
        # torch.cuda.empty_cache()
        module, best_epoch = train_loop(module, 
                            EPOCHS, 
                            train_dataset=dataset_dict['Iterators']['train'], 
                            val_dataset=dataset_dict['Iterators']['val'],
                            criterion=nn.SmoothL1Loss(reduction='mean'),
                            optimizer=optimizer,
                            lr_scheduler=scheduler,
                            batch_level_scheduler=batch_level_steps,
                            n_batches=n_batches,
                            save_per_epoch=True,
                            save_path=f"../models/{regression_head_mode}/fnv_with_{llm_name}",
                            return_best_epoch_idx=True)
        
        # Load best epoch model
        module = load_model(module, f"../models/{regression_head_mode}/fnv_with_{llm_name}_{best_epoch}.pth", DEVICE)

        # Model evaluation
        module.eval()

        predictions, true_targets = evaluate(module, 
                                            dataset_dict['Iterators']['test'], 
                                            nn.SmoothL1Loss(reduction='mean'))
        
        path_predictions = f"../predictions/{regression_head_mode}/eval_fnv_with_{llm_name}_{best_epoch}.pt"
        torch.save({'yhat':predictions, 'ytrue':true_targets}, path_predictions)
    
    else:
        module = load_model(module, module_path, DEVICE)

        # Model evaluation
        module.eval()

        predictions, true_targets = evaluate(module, 
                                            dataset_dict['Iterators']['test'], 
                                            nn.SmoothL1Loss(reduction='mean'))
        model_name = module_path[module_path.index('fnv_with'):]
        path_predictions = f"../predictions/{regression_head_mode}/eval_{model_name}"
        torch.save({'yhat':predictions, 'ytrue':true_targets}, path_predictions)

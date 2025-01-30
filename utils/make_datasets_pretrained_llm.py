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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEVICE

# Load the model
checkpoint = "almanach/camembertav2-base"

CamemBERTa = AutoModel.from_pretrained(checkpoint, use_fast=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
cls_id = tokenizer("[CLS]")['input_ids'][1]
print("\nCLS token:", cls_id)

# Load the corpus
with open(COLAB + "Descriptions_FEEDIPEDIA_ENG.txt", 'r') as f:
    FEEDIPEDIA_ENG = f.readlines()

with open(COLAB + "Descriptions_FEEDIPEDIA_FR.txt", 'r') as f:
    FEEDIPEDIA_FR = f.readlines()

with open(COLAB + "Descriptions_TableINRA2018.txt", 'r') as f:
    INRA2018 = f.readlines()

# Small function to remove trailing "\n"
def remove_trailing_n_char(string, n=2):
    return string[:-(n-1)]

# Function to apply the above function to any iterable.
def apply(func, iterable):
    return list(map(func, iterable))

# Clean our texts
INRA2018 = apply(remove_trailing_n_char, INRA2018)
FEEDIPEDIA_FR = apply(remove_trailing_n_char, FEEDIPEDIA_FR)
FEEDIPEDIA_ENG = apply(remove_trailing_n_char, FEEDIPEDIA_ENG)

ALL_TEXT = INRA2018 + FEEDIPEDIA_FR + FEEDIPEDIA_ENG
FR_TEXT = INRA2018 + FEEDIPEDIA_FR
        
TARGETS = ["UFL", "UFV", "BPR", "PDI", "PDIA"]
IN_FEATURES =  ["MM", "MAT", "CB", "NDF", "ADF", "EE"]
DB = pd.read_excel(COLAB + "TableINRA2018_AvecDescriptions.xlsx", header=0)

# For robust evaluation, we split train/val/test sets
# based on tuples of input numerical values

# List of unique tuples of input numerical values
print("\nGetting all unique IR value combinations...")
DB['ir_tuple'] = list(map(tuple, DB[IN_FEATURES].values))
unique_ir_values = DB['ir_tuple'].unique()

print("\nShuffling and splitting train/val/test sets...")
shuffled_ir_values = rd.sample(list(unique_ir_values), len(unique_ir_values))

split_70 = int(0.7 * len(unique_ir_values))
split_85 = int(0.85 * len(unique_ir_values))

train_ir_values = shuffled_ir_values[:split_70]
val_ir_values = shuffled_ir_values[split_70:split_85]
test_ir_values = shuffled_ir_values[split_85:]

print("\nCreating actual index lists for train/val/test...")
train_idx = DB.index[DB['ir_tuple'].isin(train_ir_values)].tolist()
val_idx = DB.index[DB['ir_tuple'].isin(val_ir_values)].tolist()
test_idx = DB.index[DB['ir_tuple'].isin(test_ir_values)].tolist()

# Drop the temporary column if unnecessary
DB.drop(columns=['ir_tuple'], inplace=True)

# We verify that there are no shared indices between our train and our val/test sets
assert set(test_idx).intersection(set(train_idx)) == set(), "There are shared indices."
assert set(val_idx).intersection(set(train_idx)) == set(), "There are shared indices."

print("\nAll indices are ready.")

# For better training, it is important to normalize
# We make sure there is no data leakage through normalization 
# by fitting the normalizer on the training set not modifying it afterwards
INPUT_NORM = MinMaxScaler(feature_range=(-1,1))
INPUT_NORM.fit(DB[IN_FEATURES].iloc[train_idx])

TARGET_NORM = MinMaxScaler(feature_range=(-1,1))
TARGET_NORM.fit(DB[TARGETS].iloc[train_idx])
targets_max = TARGET_NORM.data_max_
targets_min = TARGET_NORM.data_min_

# Creation of train/val/test Datasets
training_data = FourragesDataset(db=DB, 
                                dataset_idx=train_idx,
                                device=DEVICE,
                                tokenizer=tokenizer,
                                mode="with_desc")

val_data = FourragesDataset(db=DB, 
                                dataset_idx=val_idx,
                                device=DEVICE,
                                tokenizer=tokenizer,
                                mode="with_desc")

test_data = FourragesDataset(db=DB, 
                                dataset_idx=test_idx,
                                device=DEVICE,
                                tokenizer=tokenizer,
                                mode="with_desc")

# Creation of Dataloaders
BATCH = 64
train_iterator = DataLoader(training_data, 
                            batch_size=BATCH, 
                            shuffle=True, 
                            # num_workers=0,
                            collate_fn=collate_fn)

val_iterator = DataLoader(val_data, 
                            batch_size=BATCH, 
                            shuffle=True, 
                            # num_workers=0,
                            collate_fn=collate_fn)

test_iterator = DataLoader(test_data,
                            batch_size=BATCH, 
                            shuffle=True, 
                            # num_workers=0,
                            collate_fn=collate_fn)

if __name__ == "__main__":
    datasets_dico = {'Datasets':{'train':training_data, 'val':val_data, 'test':test_data},
                     'Iterators':{'train':train_iterator, 'val':val_iterator, 'test':test_iterator}}
    
    with open("./pretrained_llm_datasets.pkl", "wb") as f:
        pkl.dump(datasets_dico, f)
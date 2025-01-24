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

checkpoint = "almanach/camembertav2-base"

CamemBERTa = AutoModel.from_pretrained(checkpoint).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
cls_id = tokenizer("[CLS]")['input_ids'][1]
print("\nCLS token:", cls_id)

with open(COLAB + "Descriptions_FEEDIPEDIA_ENG.txt", 'r') as f:
    FEEDIPEDIA_ENG = f.readlines()

with open(COLAB + "Descriptions_FEEDIPEDIA_FR.txt", 'r') as f:
    FEEDIPEDIA_FR = f.readlines()

with open(COLAB + "Descriptions_TableINRA2018.txt", 'r') as f:
    INRA2018 = f.readlines()

def remove_trailing_n_char(string, n=2):
    return string[:-(n-1)]

def apply(func, iterable):
    return list(map(func, iterable))

INRA2018 = apply(remove_trailing_n_char, INRA2018)
FEEDIPEDIA_FR = apply(remove_trailing_n_char, FEEDIPEDIA_FR)
FEEDIPEDIA_ENG = apply(remove_trailing_n_char, FEEDIPEDIA_ENG)

ALL_TEXT = INRA2018 + FEEDIPEDIA_FR + FEEDIPEDIA_ENG
FR_TEXT = INRA2018 + FEEDIPEDIA_FR

# idx = int(torch.randint(len(FR_TEXT), size=(1,1))[0])
# INPUT = FR_TEXT[idx:idx+10]
# TK_INPUT = tokenizer(INPUT)

def rdy_input(text, tokenizer):
    """
    Function to add padding for consistent size
    after tokenization of input.
    """
    pad_id = tokenizer("[PAD]")['input_ids'][1]
    tk_input = tokenizer(text)
    tk_input_emb_ids, tk_input_mask = tk_input['input_ids'], tk_input['attention_mask']

    maxlen = np.max([len(desc) for desc in tk_input_emb_ids])
    n_to_fill = [maxlen - len(desc) for desc in tk_input_emb_ids]
    tk_input_emb_ids = [desc[:-1] + [pad_id]*n_to_fill[i] + [desc[-1]] for i, desc in enumerate(tk_input_emb_ids)]
    return tk_input_emb_ids, tk_input_mask, n_to_fill

def rdy_output(output, n_to_fill):
    # car output de taille Batch x MaxLen x EmbSize
    output = output.last_hidden_state
    maxlen = output.size()[1]
    return torch.stack([output[i, maxlen - n_missing - 1] for i, n_missing in enumerate(n_to_fill)])

# print("\nInput Text:", INPUT)

# TK_INPUT_EMB, TK_INPUT_MASK, N_TO_FILL = rdy_input(INPUT, tokenizer)

# print("\nTokenized input:", TK_INPUT_EMB)

# outputs = CamemBERTa(torch.tensor(TK_INPUT_EMB, device=device)) #torch.tensor(feature_extractor(INPUT, return_tensors="pt"), device=device)

# outputs = rdy_output(outputs, N_TO_FILL)
# print("\nFeatures :", outputs.size())


# Fodder Nutritional Value Model (ou Model de Valeurs Nutritionnelles de Fourrages)
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
class FNVModel(nn.Module):
    def __init__(self):
        super(FNVModel, self).__init__()
        self.history = {"epochs":[], "test":[]}
        self.mode = 'with_desc'

        self.feature_extractor = CamemBERTa
        for parameters in self.feature_extractor.parameters():
          parameters.to(DEVICE)

        EMB_SIZE = 768
        
        self.repr_enricher = nn.Sequential(nn.Linear(6 + EMB_SIZE, 2048),
                                    #    nn.LayerNorm(2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, 1024),
                                    #    nn.LayerNorm(512),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(1024, 512),
                                       nn.ReLU())
        
        self.prot_regressor = nn.Sequential(nn.Linear(512 + 6, 512),
                                    #    nn.LayerNorm(2048),
                                       nn.ReLU(),
                                       nn.Linear(512, 256),
                                    #    nn.LayerNorm(512),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(256, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 3))
        
        self.en_regressor = nn.Sequential(nn.Linear(512 + 6, 512),
                                    #    nn.LayerNorm(2048),
                                       nn.ReLU(),
                                       nn.Linear(512, 256),
                                    #    nn.LayerNorm(512),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(256, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 2))
        for block in [self.repr_enricher, 
                      self.prot_regressor,
                      self.en_regressor]:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                for parameters in layer.parameters():
                  parameters.to(DEVICE)

    def rdy_output(self, output, n_to_fill, cls_id=None):
        # car output de taille Batch x MaxLen x EmbSize
        output = output.last_hidden_state
        maxlen = output.size()[1]
        if cls_id is not None:
          return output[:, 1]
        return torch.stack([output[i, maxlen - n_missing - 1] for i, n_missing in enumerate(n_to_fill)])

    def forward(self, x, desc, n_to_fill):
        # compute features
        text_features = self.feature_extractor(torch.tensor(desc,
                                                            dtype=torch.int,
                                                            device=DEVICE))

        # prepare features for regression task
        text_features = self.rdy_output(text_features, 
                                        n_to_fill, 
                                        cls_id=cls_id).to(DEVICE)

        # enrich representations
        regression_emb = self.repr_enricher(torch.cat((text_features, x), dim=1))
        prot_output = self.prot_regressor(torch.cat((regression_emb, x), dim=1)) # skip connection
        en_output = self.en_regressor(torch.cat((regression_emb, x), dim=1))
        
        return torch.cat((en_output, prot_output), dim=1)

    def train_log(self, train_batch_losses, val_batch_losses, train_loss, validation_loss):
        self.history["epochs"].append({"train_batch_losses":train_batch_losses, 
                                "val_batch_losses":val_batch_losses, 
                                "train_loss":train_loss, 
                                "validation_loss":validation_loss})
    
    def test_log(self, test_batch_losses, test_loss):
        self.history["test"].append({"test_batch_losses":test_batch_losses,
                                "test_loss":test_loss})
    
    def save_model(self, path):
        """
        Saves the model architecture and state using state-of-the-art PyTorch methods.

        Parameters:
            path (str): The path to save the model file.
        """
        # Save model state dictionary and any additional information like architecture, optimizer, or history
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__,
            'history': self.history
        }, path)
        print(f"Model saved successfully at: {path}")
        
TARGETS = ["UFL", "UFV", "BPR", "PDI", "PDIA"]
IN_FEATURES =  ["MM", "MAT", "CB", "NDF", "ADF", "EE"]
DB = pd.read_excel(COLAB + "TableINRA2018_AvecDescriptions.xlsx", header=0)

# Pour une évaluation robuste, on sépare train/val/test 
# en fonction des n_uplets des valeurs infrarouges de chaque fourrage

# Liste des n-uplets uniques de valeurs infrarouge d'entrée
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

assert set(test_idx).intersection(set(train_idx)) == set(), "There are shared indices."
assert set(val_idx).intersection(set(train_idx)) == set(), "There are shared indices."

print("\nAll indices are ready.")

INPUT_NORM = MinMaxScaler(feature_range=(-1,1))
INPUT_NORM.fit(DB[IN_FEATURES].iloc[train_idx])

TARGET_NORM = MinMaxScaler(feature_range=(-1,1))
TARGET_NORM.fit(DB[TARGETS].iloc[train_idx])
targets_max = TARGET_NORM.data_max_
targets_min = TARGET_NORM.data_min_

# print("Training indices Size:", len(train_idx))
training_data = FourragesDataset(db=DB, 
                                dataset_idx=train_idx,
                                device=DEVICE,
                                tokenizer=tokenizer,
                                mode="with_desc")

# print("Validation indices Size:", len(val_idx))
val_data = FourragesDataset(db=DB, 
                                dataset_idx=val_idx,
                                device=DEVICE,
                                tokenizer=tokenizer,
                                mode="with_desc")

# print("Test indices Size:", len(test_idx))
test_data = FourragesDataset(db=DB, 
                                dataset_idx=test_idx,
                                device=DEVICE,
                                tokenizer=tokenizer,
                                mode="with_desc")
BATCH = 64
train_iterator = DataLoader(training_data, 
                            batch_size=BATCH, 
                            shuffle=True, 
                            # num_workers=0,
                            collate_fn=collate_fn)

# print("Training Iterator Size:", len(train_iterator))
val_iterator = DataLoader(val_data, 
                            batch_size=BATCH, 
                            shuffle=True, 
                            # num_workers=0,
                            collate_fn=collate_fn)

# print("Validation Iterator Size:", len(train_iterator))
test_iterator = DataLoader(test_data,
                            batch_size=BATCH, 
                            shuffle=True, 
                            # num_workers=0,
                            collate_fn=collate_fn)

# torch.set_num_threads(6)

def load_model(path, device):
    """
    Loads the saved model and returns it for inference.

    Parameters:
        path (str): Path to the saved model file.
        device (torch.device): The device to map the model to (e.g., 'cuda' or 'cpu').
        
    Returns:
        model: Loaded FNVModel instance ready for inference.
    """
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device)
    
    # Initialize the model architecture
    model_class = checkpoint['model_class']
    model = model_class().to(device)
    
    # Load state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully.")
    return model

class LogCoshLoss(nn.Module):
    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)

class CombinedRegressionLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.2, gamma=0.6, max_error_beta=10):
        """
        Custom loss combining MAE, MSE, and a smooth approximation of Max Error.
        Args:
            alpha (float): Weight for MAE component.
            beta (float): Weight for MSE component.
            gamma (float): Weight for Max Error component.
            max_error_beta (float): Smoothing parameter for Max Error approximation.
        """
        super(CombinedRegressionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_error_beta = max_error_beta
    
    def forward(self, y_pred, y_true):
        # MAE Component
        mae_loss = torch.mean(torch.abs(y_pred - y_true))
        
        # MSE Component
        mse_loss = torch.mean((y_pred - y_true) ** 2)
        
        # Smooth Max Error using LogSumExp Approximation
        max_error_loss = (1 / self.max_error_beta) * torch.log(
            torch.sum(torch.exp(self.max_error_beta * torch.abs(y_pred - y_true)))
        )
        
        # Combined Loss
        combined_loss = (self.alpha * mae_loss + 
                         self.beta * mse_loss + 
                         self.gamma * max_error_loss)
        
        return combined_loss


if __name__ == "__main__":
    EPOCHS = 5
    LR = 5e-4

    network = FNVModel().to(DEVICE)
    
    optimizer = torch.optim.Adam(network.parameters(), lr=LR, eps=5e-8)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                      gamma=0.97)

    module = train_loop(network, 
                        EPOCHS, 
                        train_dataset=train_iterator, 
                        val_dataset=val_iterator,
                        criterion=nn.SmoothL1Loss(reduction='mean'),
                        optimizer=optimizer,
                        lr_scheduler=scheduler)

    save_path = COLAB + "FNV_camemBERTaV2.pth"
    module.save_model(save_path)

    module = load_model(save_path, DEVICE)
    module.eval()

    predictions, true_targets = evaluate(module, 
                                         test_iterator, 
                                         nn.SmoothL1Loss(reduction='mean'))
    show_predictions(predictions, 
                      true_targets, 
                      TARGETS, 
                      target_normalizer=None, #TARGET_NORM, 
                      save_path=COLAB + "camemberta_model_test.pdf")
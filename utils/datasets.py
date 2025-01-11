# Seed for reproducibility
SEED = 62

DEVICE = 'cpu'

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

TARGETS = ["UFL", "UFV", "BPR", "PDI", "PDIA"]
IN_FEATURES =  ["MM", "MAT", "CB", "NDF", "ADF", "EE"]
class FourragesDataset(Dataset):
    def __init__(self, mode="num_only",  
                 dataset_idx=None,
                 db=None,
                 input_normalizer=None,
                 target_normalizer=None, 
                 target_cols=TARGETS,
                 num_cols=IN_FEATURES,
                 device=None):
        self.device = device
        self.dataset_idx = dataset_idx
        self.db = db.iloc[self.dataset_idx, :]

        self.target_cols = target_cols
        self.input_features = num_cols

        self.X = self.db[self.input_features].copy()
        self.X = self.X.dropna()

        self.y = self.db[self.target_cols].filter(items=self.X.index, axis=0)


        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer

        self.mode = mode
        pass

    def __getitem__(self, idx):
        x = self.X.iloc[idx:idx+1,:]
        y = self.y.iloc[idx:idx+1,:]

        # if self.input_normalizer:
        #     try: 
        #         check_is_fitted(self.input_normalizer)
        #         x = self.input_normalizer.transform(x)
        #     except:
        #         x = self.input_normalizer.fit_transform(x)
    
        # if self.target_normalizer:
        #     try:
        #         check_is_fitted(self.target_normalizer)
        #         y = self.target_normalizer.transform(y)
        #     except:
        #         y = self.target_normalizer.fit_transform(y)
        return x, y 
    
    def __len__(self):
        return len(self.X)

DB = pd.read_excel("././INRA2018_TablesFourrages_etude_prediction_20241121.xlsx", header=1)

shuffled_idx = rd.sample([i for i in range(DB.shape[0])], len([i for i in range(DB.shape[0])]))

train_idx = rd.sample(shuffled_idx, int(0.7 * DB.shape[0]))
val_idx = rd.sample(list(set(shuffled_idx).difference(set(train_idx))), int(0.15 * DB.shape[0]))
test_idx = list(set(shuffled_idx).difference(set(val_idx + train_idx)))

assert set(test_idx).intersection(set(train_idx)) == set(), "There are shared indices."
assert set(val_idx).intersection(set(train_idx)) == set(), "There are shared indices."

INPUT_NORM = MinMaxScaler(feature_range=(-1,1))
INPUT_NORM.fit(DB[IN_FEATURES].iloc[train_idx])

TARGET_NORM = MinMaxScaler(feature_range=(-1,1))
TARGET_NORM.fit(DB[TARGETS].iloc[train_idx])
targets_max = TARGET_NORM.data_max_
targets_min = TARGET_NORM.data_min_

def collate_fn(data):
    x = pd.concat([sample[0] for sample in data], axis=0)
    y = pd.concat([sample[1] for sample in data], axis=0)

    x = torch.tensor(INPUT_NORM.transform(x), 
                              device=DEVICE,
                              requires_grad=True).to(torch.float32)
    y = torch.tensor(TARGET_NORM.transform(y), 
                              device=DEVICE).to(torch.float32)
    return x, y

# print("Training indices Size:", len(train_idx))
training_data = FourragesDataset(db=DB, 
                                dataset_idx=train_idx,
                                device=DEVICE)

# print("Validation indices Size:", len(val_idx))
val_data = FourragesDataset(db=DB, 
                                dataset_idx=val_idx,
                                device=DEVICE)

# print("Test indices Size:", len(test_idx))
test_data = FourragesDataset(db=DB, 
                                dataset_idx=test_idx,
                                device=DEVICE)

train_iterator = DataLoader(training_data, 
                            batch_size=32, 
                            shuffle=True, 
                            collate_fn=collate_fn)

# print("Training Iterator Size:", len(train_iterator))
val_iterator = DataLoader(val_data, 
                            batch_size=32, 
                            shuffle=True, 
                            collate_fn=collate_fn)

# print("Validation Iterator Size:", len(train_iterator))
test_iterator = DataLoader(test_data,
                            batch_size=32, 
                            shuffle=True, 
                            collate_fn=collate_fn)
# print("Test Iterator Size:", len(train_iterator))


class NotreModele(nn.Module):
    def __init__(self):
        super(NotreModele, self).__init__()
        self.history = {"epochs":[], "test":[]}
        self.regressor = nn.Sequential(nn.Linear(6,2048),
                                    #    nn.LayerNorm(2048),
                                       nn.ReLU(),
                                       nn.Linear(2048, 512),
                                    #    nn.LayerNorm(512),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(512,5))

        for layer in self.regressor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.regressor(x)

    def train_log(self, train_batch_losses, val_batch_losses, train_loss, validation_loss):
        self.history["epochs"].append({"train_batch_losses":train_batch_losses, 
                                "val_batch_losses":val_batch_losses, 
                                "train_loss":train_loss, 
                                "validation_loss":validation_loss})
    
    def test_log(self, test_batch_losses, test_loss):
        self.history["test"].append({"test_batch_losses":test_batch_losses,
                                "test_loss":test_loss})
    
def train_step(module, batch, batch_idx, criterion, optimizer):
    X, y = batch
    module.train(True)
    
    model_outputs = module(X)
    loss = 0
    for i in range(y.size(-1)):
        loss += criterion(model_outputs[:,i], y[:,i])
    loss = loss / y.size(-1)
    # print(f"\n\033[1;37mBatch loss {batch_idx+1} : {loss.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    return module, loss

def eval_step(module, batch, batch_idx, criterion, optimizer=None, training=True):
    X, y = batch
    with torch.no_grad():
        model_outputs = module(X)
        loss = 0
        for i in range(y.size(-1)):
            loss += criterion(model_outputs[:,i], y[:,i])
        loss = loss / y.size(-1)
        if training:
            # print(f"\n\033[1;32mValidation Batch loss {batch_idx+1} : {loss.item()}")
            return module, loss
        else:
            # print(f"\n\033[1;32mTest Batch loss {batch_idx+1} : {loss.item()}")
            return module, loss, model_outputs, y

    


def train_loop(module, EPOCHS, train_dataset, val_dataset, criterion, optimizer):
    for epoch in range(EPOCHS):
        module.train(True)
        train_batch_losses = []
        for batch_idx in range(len(train_dataset)):
            batch = next(iter(train_dataset))
            # print(batch[0].size(), batch[1].size())
            module, loss = train_step(module, batch, batch_idx, criterion, optimizer)
            train_batch_losses.append(loss.item())
        train_loss = np.mean(train_batch_losses)

        module.train(False)
        val_batch_losses = []
        for batch_idx in range(len(val_dataset)):
            batch = next(iter(val_dataset))
            module, loss = eval_step(module, batch, batch_idx, criterion)
            val_batch_losses.append(loss.item())
        val_loss = np.mean(val_batch_losses)

        module.train_log(train_batch_losses, val_batch_losses, train_loss, val_loss)
        print(f"\n\033[1;33mEpoch {epoch+1} :\n\033[1;37mTraining Loss : {train_loss}")
        print(f"\033[1;32mValidation Loss : {val_loss}")
    return module

def evaluate(module, test_dataset, criterion):
    module.train(False)
    test_batch_losses = []
    predictions = []
    true_targets = []
    for batch_idx in range(len(test_dataset)):
        batch = next(iter(test_dataset))
        module, loss, y_preds, y = eval_step(module, batch, batch_idx, criterion, training=False)

        test_batch_losses.append(loss.item())
        predictions.append(y_preds)
        true_targets.append(y)

    test_loss = np.mean(test_batch_losses)
    module.test_log(test_batch_losses, test_loss)
    print(f"\nTest Loss : {test_loss}")
    return predictions, true_targets

def show_predictions(predictions, true_targets, target_names, save=True):
    y_hat = torch.cat(predictions, dim=0).numpy()
    y_true = torch.cat(true_targets, dim=0).numpy()

    fig, ax = plt.subplots(len(target_names),1, figsize=(6,33))
    for i, target in enumerate(target_names):
        Y_HAT = y_hat[:,i]
        Y_TRUE = y_true[:,i]
        r2 = r2_score(Y_TRUE, Y_HAT)
        rmse = root_mean_squared_error(Y_TRUE, Y_HAT)
        mae = mean_absolute_error(Y_TRUE, Y_HAT)
        mape = mean_absolute_percentage_error(Y_TRUE, Y_HAT)
        plt.subplot(len(target_names), 1, i+1)
        sns.scatterplot(x=Y_HAT, y=Y_TRUE, 
                 palette='viridis', 
                 hue=np.abs(Y_HAT - Y_TRUE))
        sns.lineplot(x=Y_TRUE, y=Y_TRUE)
        plt.title(f'Model Predictions vs Ground Truth for {target}\nRMSE: {rmse:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}, MAPE : {mape:.4f}')
        plt.xlabel(f'Predicted {target}')
        plt.ylabel(f'True {target}')
    if save:
        plt.savefig(f"./model_test.pdf", format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    EPOCHS = 100
    LR = 1e-4

    network = NotreModele()
    for tensor in network.parameters():
        torch.nn.utils.clip_grad_norm_(tensor, 1.0)
    optimizer = torch.optim.Adam(network.parameters(), lr=LR, eps=5e-8)
    module = train_loop(network, 
                        EPOCHS, 
                        train_dataset=train_iterator, 
                        val_dataset=val_iterator,
                        criterion=nn.SmoothL1Loss(reduction='mean'),
                        optimizer=optimizer)
    predictions, true_targets = evaluate(module, 
                                         test_iterator, 
                                         nn.SmoothL1Loss(reduction='mean'))
    show_predictions(predictions, true_targets, TARGETS)


    
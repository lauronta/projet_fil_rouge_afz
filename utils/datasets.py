# Seed for reproducibility
SEED = 62

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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLAB = "/content/drive/MyDrive/Projet_Fil_Rouge_AFZ/camembertaV2/"

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
                 tokenizer=None,
                 padding_token="[PAD]",
                 cls_token="[CLS]",
                 device=None):
        self.device = device
        self.mode = mode
        if self.mode != "num_only":
            assert tokenizer is not None, "tokenizer argument must be specified when not in 'num_only' mode."
            self.tokenizer = tokenizer
            self.pad_id = self.tokenizer(padding_token)['input_ids'][1]
            self.cls_id = self.tokenizer(cls_token)['input_ids'][1]

        self.dataset_idx = dataset_idx
        self.db = db.iloc[self.dataset_idx, :]

        self.target_cols = target_cols
        self.input_features = num_cols

        self.X = self.db[self.input_features].copy()

        if self.mode != "num_only":
            self.desc = self.db["Descriptions"].copy()
            combined = pd.concat([self.X, self.desc], axis=1).dropna()
            self.X = combined[self.input_features]
            self.desc = combined["Descriptions"] if self.mode != "num_only" else None
            assert np.sum(pd.isna(self.desc)) == 0, "There are NA's !!"

            print("\nTokenizing dataset..")
            self.tk_input_emb_ids, self.tk_input_mask, self.n_to_fill = self.rdy_input(self.desc)
            self.desc.iloc[:] = self.tk_input_emb_ids
            self.n_to_fill = pd.DataFrame(self.n_to_fill, columns=["N_to_fill"], index=self.desc.index)
            print("\nDataset tokenized.")
        else:
            self.X = self.X.dropna()
            self.desc = None

        assert np.sum(np.sum(pd.isna(self.X))) == 0, "There are NA's !!"

        self.y = self.db[self.target_cols].filter(items=self.X.index, axis=0)

        self.input_normalizer = input_normalizer
        self.target_normalizer = target_normalizer

        self.mode = mode
        pass
    
    def rdy_input(self, text):
        """
        Function to add padding for consistent size
        after tokenization of input.
        """
        if isinstance(text, pd.DataFrame) or isinstance(text, pd.Series):
            text = list(text)
        if isinstance(text, torch.Tensor):
            text = list(text)
        tk_input = self.tokenizer(text)
        tk_input_emb_ids, tk_input_mask = tk_input['input_ids'], tk_input['attention_mask']

        maxlen = np.max([len(desc) for desc in tk_input_emb_ids])
        n_to_fill = [maxlen - len(desc) for desc in tk_input_emb_ids]
        tk_input_emb_ids = [desc[:-1] + [self.pad_id]*n_to_fill[i] + [desc[-1]] for i, desc in enumerate(tk_input_emb_ids)]
        return tk_input_emb_ids, tk_input_mask, n_to_fill
    
    def __getitem__(self, idx):
        x = self.X.iloc[idx:idx+1,:]
        y = self.y.iloc[idx:idx+1,:]

        if self.desc is not None:
            desc = self.desc.iloc[idx:idx+1]
            n_to_fill = self.n_to_fill.iloc[idx:idx+1]
            return x, desc, n_to_fill, y
        return x, y 
    
    def __len__(self):
        return len(self.X)

DB = pd.read_excel(COLAB + "INRA2018_TablesFourrages_etude_prediction_20241121.xlsx", header=1)

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
    if len(data[0]) == 2: # Si on reçoit (x,y)
        x = pd.concat([sample[0] for sample in data], axis=0)
        y = pd.concat([sample[1] for sample in data], axis=0)
        desc = None
        n_to_fill = None
    else: # si on reçoit (x, desc, n_to_fill, y)
        x = pd.concat([sample[0] for sample in data], axis=0)
        desc = list(pd.concat([sample[1] for sample in data], axis=0))
        n_to_fill = list(pd.concat([sample[2].iloc[0] for sample in data], axis=0))
        y = pd.concat([sample[3] for sample in data], axis=0)

    x = torch.tensor(INPUT_NORM.transform(x), 
                              device=DEVICE,
                              requires_grad=True).to(torch.float32)
    y = torch.tensor(TARGET_NORM.transform(y), 
                              device=DEVICE).to(torch.float32)
    if desc is not None:
        return x, desc, n_to_fill, y
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


def save_model_func(model, save_path, tag=""):
        """
        Saves the model architecture and state using state-of-the-art PyTorch methods.

        Parameters:
            path (str): The path to save the model file.
        """
        # Save model state dictionary and any additional information like architecture, optimizer, or history
        torch.save({'model_state_dict': model.state_dict()}, 
                   tag + "_" + save_path)
        print(f"Model saved successfully at: {tag + '_' + save_path}")

def load_model(model, save_path, device, tag=""):
        """
        Loads the saved model and returns it for inference.

        Parameters:
            path (str): Path to the saved model file.
            device (torch.device): The device to map the model to (e.g., 'cuda' or 'cpu').
            
        Returns:
            model: Loaded FNVModel instance ready for inference.
        """
        # Load checkpoint
        checkpoint = torch.load(tag + "_" + save_path, map_location=device)
        

        # Load state dictionary
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print("Model loaded successfully.")
        return model

def train_step(module, 
                batch, 
                batch_idx, 
                criterion, 
                optimizer):
    module.train(True)
    if module.mode == 'num_only':
        X, y = batch
        model_outputs = module(X)
    else:
        X, desc, n_to_fill, y = batch
        model_outputs = module(X, desc, n_to_fill)
    loss = 0
    for i in range(y.size(-1)):
        loss += criterion(model_outputs[:,i], y[:,i])
    loss = loss / y.size(-1)
    print(f"\n\033[1;37mBatch loss {batch_idx+1} : {loss.item()}")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    return module, loss

def eval_step(module, batch, batch_idx, criterion, optimizer=None, training=True):
    with torch.no_grad():
        if module.mode == 'num_only':
            X, y = batch
            model_outputs = module(X)
        else:
            X, desc, n_to_fill, y = batch
            model_outputs = module(X, desc, n_to_fill)
        loss = 0
        for i in range(y.size(-1)):
            loss += criterion(model_outputs[:,i], y[:,i])
        loss = loss / y.size(-1)
        if training:
            print(f"\n\033[1;32mValidation Batch loss {batch_idx+1} : {loss.item()}")
            return module, loss
        else:
            print(f"\n\033[1;32mTest Batch loss {batch_idx+1} : {loss.item()}")
            return module, loss, model_outputs, y

def train_loop(module, 
              EPOCHS, 
              train_dataset, 
              val_dataset, 
              criterion, 
              optimizer,
              lr_scheduler=None):
    for epoch in range(EPOCHS):
        module.train(True)
        train_batch_losses = []
        for batch_idx in range(len(train_dataset)):
            batch = next(iter(train_dataset))
            # print(batch[0].size(), batch[1].size())
            module, loss = train_step(module, 
                                      batch, 
                                      batch_idx, 
                                      criterion, 
                                      optimizer)
            train_batch_losses.append(loss.item())
        if lr_scheduler is not None:
          lr_scheduler.step()
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

def show_predictions(predictions, true_targets, target_names, target_normalizer=None, save=True, save_path="./model_test.pdf"):
    if target_normalizer is not None:
        y_hat = target_normalizer.inverse_transform(torch.cat(predictions, dim=0).cpu().numpy())
        y_true = target_normalizer.inverse_transform(torch.cat(true_targets, dim=0).cpu().numpy())
    else:
        y_hat = torch.cat(predictions, dim=0).cpu().numpy()
        y_true = torch.cat(true_targets, dim=0).cpu().numpy()

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
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    EPOCHS = 100
    LR = 1e-4

    network = NotreModele()
    
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


    
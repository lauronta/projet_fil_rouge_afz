# Seed for reproducibility
SEED = 62

# Classic imports
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

import random as rd
rd.seed(SEED)
import numpy as np
np.random.seed(SEED)
import pandas as pd
import pickle as pkl
from functools import partial

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
PATH_TO_DB = "../../INRA2018_TablesFourrages_etude_prediction_20241121.xlsx" #/content/drive/MyDrive/Projet_Fil_Rouge_AFZ/camembertaV2/INRA2018_TablesFourrages_etude_prediction_20241121.xlsx"

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
                 device=None):
        self.device = device
        self.mode = mode
        if self.mode != "num_only":
            assert tokenizer is not None, "tokenizer argument must be specified when not in 'num_only' mode."
            self.tokenizer = tokenizer
            padding_token = tokenizer.pad_token   # e.g., BERT typically shows "[PAD]"
            cls_token = tokenizer.cls_token
            self.pad_id = tokenizer.pad_token_id
            self.cls_id = tokenizer.cls_token_id

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

def collate_fn(data, input_norm, target_norm, device):
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

    x = torch.tensor(input_norm.transform(x), 
                            device=device,
                            requires_grad=True).to(torch.float32)
    y = torch.tensor(target_norm.transform(y), 
                            device=device).to(torch.float32)
    if desc is not None:
        return x, desc, n_to_fill, y
    return x, y

def save_model_func(model, save_path, tag=""):
        """
        Saves the model architecture and state using state-of-the-art PyTorch methods.

        Parameters:
            path (str): The path to save the model file.
        """
        # Save model state dictionary and any additional information like architecture, optimizer, or history
        if tag == "":
            torch.save({'model_state_dict': model.state_dict()}, 
                      save_path)
            print(f"Model saved successfully at: {save_path}")
        else:
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
        if tag == "":
            checkpoint = torch.load(save_path, map_location=device)
        else:
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
    """
    Performs a single training step.
    
    Args:
        module: The model being trained.
        batch: A batch of data.
        batch_idx: Index of the batch.
        criterion: Loss function.
        optimizer: Optimization algorithm.

    Returns:
        Updated module and computed loss.
    """
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
    """
    Performs an evaluation step.
    
    Args:
        module: The model being evaluated.
        batch: A batch of data.
        batch_idx: Index of the batch.
        criterion: Loss function.
        training: Boolean indicating whether it's training or testing.

    Returns:
        Module, loss, and optionally model outputs and targets.
    """
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
              lr_scheduler=None,
              batch_level_scheduler=False,
              n_batches=1,
              save_per_epoch=False,
              save_path=None,
              return_best_epoch_idx=False):
    """
    Executes the full training loop.
    
    Args:
        module: Model to train.
        EPOCHS: Number of training epochs.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        lr_scheduler: Learning rate scheduler (optional).

    Returns:
        Trained model.
    """
    best_val_loss = 0
    best_epoch = 0
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
            if lr_scheduler is not None and batch_level_scheduler:
                if batch_idx % n_batches == 0:
                    print("New LR:", lr_scheduler.get_last_lr()[0])
                    lr_scheduler.step()
        if lr_scheduler is not None and not batch_level_scheduler:
            print(f"Epoch {epoch} LR:", lr_scheduler.get_last_lr()[0])
            lr_scheduler.step()
        train_loss = np.mean(train_batch_losses)

        module.train(False)
        val_batch_losses = []
        for batch_idx in range(len(val_dataset)):
            batch = next(iter(val_dataset))
            module, loss = eval_step(module, batch, batch_idx, criterion)
            val_batch_losses.append(loss.item())
        val_loss = np.mean(val_batch_losses)

        if epoch == 0:
            best_val_loss = val_loss
            best_epoch = epoch + 1
        else:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1

        module.train_log(train_batch_losses, val_batch_losses, train_loss, val_loss)
        print(f"\n\033[1;33mEpoch {epoch+1} :\n\033[1;37mTraining Loss : {train_loss}")
        print(f"\033[1;32mValidation Loss : {val_loss}")
        if save_per_epoch:
            if save_path is not None:
                save_path_epoch = save_path + f"_{epoch + 1}.pth"
                module.save_model(save_path_epoch)
            else:
                raise ValueError("save_per_epoch was set to True but save_path is None. You must specify a save_path.")
    if return_best_epoch_idx:
        return module, best_epoch
    return module

def evaluate(module, test_dataset, criterion):
    """
    Evaluates the model on the test dataset.
    
    Args:
        module: Trained model.
        test_dataset: Test dataset.
        criterion: Loss function.

    Returns:
        Predictions and true targets.
    """
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

def load_datasets(path):
    with open(path, "rb") as f:
        dataset_dico = pkl.load(f)
    return dataset_dico

if __name__ == "__main__":
    DB = pd.read_excel(PATH_TO_DB, header=1)

    # For robust evaluation, we split train/val/test sets
    shuffled_idx = rd.sample([i for i in range(DB.shape[0])], len([i for i in range(DB.shape[0])]))

    train_idx = rd.sample(shuffled_idx, int(0.7 * DB.shape[0]))
    val_idx = rd.sample(list(set(shuffled_idx).difference(set(train_idx))), int(0.15 * DB.shape[0]))
    test_idx = list(set(shuffled_idx).difference(set(val_idx + train_idx)))

    # We verify that there are no shared indices between our train and our val/test sets
    assert set(test_idx).intersection(set(train_idx)) == set(), "There are shared indices."
    assert set(val_idx).intersection(set(train_idx)) == set(), "There are shared indices."

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
                                    device=DEVICE)

    val_data = FourragesDataset(db=DB, 
                                    dataset_idx=val_idx,
                                    device=DEVICE)

    test_data = FourragesDataset(db=DB, 
                                    dataset_idx=test_idx,
                                    device=DEVICE)

    custom_collate_fn = partial(collate_fn, input_norm=INPUT_NORM, target_norm=TARGET_NORM, device=DEVICE)
    # Creation of Dataloaders
    train_iterator = DataLoader(training_data, 
                                batch_size=32, 
                                shuffle=True, 
                                collate_fn=custom_collate_fn)

    val_iterator = DataLoader(val_data, 
                                batch_size=32, 
                                shuffle=True, 
                                collate_fn=custom_collate_fn)

    test_iterator = DataLoader(test_data,
                                batch_size=32, 
                                shuffle=True, 
                                collate_fn=custom_collate_fn)

    datasets_dico = {'Datasets':{'train':training_data, 'val':val_data, 'test':test_data},
                     'Iterators':{'train':train_iterator, 'val':val_iterator, 'test':test_iterator},
                     'Normalizer':{'input':INPUT_NORM, 'target':TARGET_NORM}}
    
    with open("../test_num_only_datasets.pkl", "wb") as f:
        pkl.dump(datasets_dico, f)


    
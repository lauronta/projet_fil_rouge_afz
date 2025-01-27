# Seed for reproducibility
SEED = 62

# Classic imports
import os
import random as rd
rd.seed(SEED)
import numpy as np
np.random.seed(SEED)

# PyTorch imports
import torch
torch.manual_seed(SEED)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
torch.autograd.set_detect_anomaly(True)

class BaseModel():
    """
    Classe de base permettant d'ajouter n'importe quelle méthode à tous
    nos modèles qui hériterons de cette classe.
    """
    def __init__(self):
        self.history = {"epochs":[], "test":[]}
    
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
        
class ModeleSansDescription(nn.Module, BaseModel):
    """
    Simple MLP Model that only receives as input : MS, MM, MAT, CB, NDF, ADF, EE
    """
    def __init__(self):
        super().__init__()
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
    

class FNVModel(nn.Module, BaseModel):
    """
    This model uses a pre-trained LLM to process fodder descriptions and extracts 
    embeddings representing these descriptions and combines them with
    numerical input values to make its final predictions.
    """
    def __init__(self, llm, device, cls_id=None):
        """
        llm : LLM model from a call of AutoModel.from_pretrained(checkpoint)
        device : Device on which to store the model
        """
        super(FNVModel, self).__init__()

        self.device = device
        self.cls_id = cls_id

        self.feature_extractor = llm.to(self.device)
        for parameters in self.feature_extractor.parameters():
          parameters.to(self.device)

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
        
        # Initialisation des paramètres
        for block in [self.repr_enricher, 
                      self.prot_regressor,
                      self.en_regressor]:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                for parameters in layer.parameters():
                  parameters.to(self.device)

    def rdy_output(self, output, n_to_fill, cls_id=None):
        """
        Process LLM output to return last true positions instead of padding.
        """
        # Output is of size:  Batch x MaxLen x EmbSize
        output = output.last_hidden_state
        maxlen = output.size()[1]
        if cls_id is not None:
          return output[:, 1]
        return torch.stack([output[i, maxlen - n_missing - 1] for i, n_missing in enumerate(n_to_fill)])

    def forward(self, x, desc, n_to_fill):
        # compute features
        text_features = self.feature_extractor(torch.tensor(desc,
                                                            dtype=torch.int,
                                                            device=self.device))

        # prepare features for regression task
        text_features = self.rdy_output(text_features, 
                                        n_to_fill, 
                                        cls_id=self.cls_id).to(self.device)

        # enrich representations
        regression_emb = self.repr_enricher(torch.cat((text_features, x), dim=1))
        prot_output = self.prot_regressor(torch.cat((regression_emb, x), dim=1)) # skip connection
        en_output = self.en_regressor(torch.cat((regression_emb, x), dim=1))
        
        return torch.cat((en_output, prot_output), dim=1)

if __name__ == "__main__":
    model = ModeleSansDescription()
    print(model.history)
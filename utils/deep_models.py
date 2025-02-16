# Seed for reproducibility
SEED = 62

# Classic imports
import os
import random as rd
rd.seed(SEED)
import numpy as np
np.random.seed(SEED)
from typing import List, Optional, Callable

# PyTorch imports
import torch
torch.manual_seed(SEED)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
torch.autograd.set_detect_anomaly(True)

class BaseModel(nn.Module):
    """
    Classe de base permettant d'ajouter n'importe quelle méthode à tous
    nos modèles qui hériterons de cette classe.
    """
    def __init__(self):
        super().__init__()
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
        state_dict = self.state_dict()
        torch.save({
            'model_state_dict': state_dict,
            'model_class': self.__class__,
            'history': self.history
        }, path)
        print(f"Model saved successfully at: {path}")
        
class ModeleSansDescription(BaseModel):
    """
    Simple MLP Model that only receives as input : MS, MM, MAT, CB, NDF, ADF, EE
    """
    def __init__(self):
        super().__init__()
        self.mode = 'num_only'
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
    
# Fodder Nutritional Value Model (ou Model de Valeurs Nutritionnelles de Fourrages)
class FNVModel(BaseModel):
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
        super().__init__()

        self.device = device
        self.cls_id = cls_id

        self.feature_extractor = llm.to(self.device)
        for parameters in self.feature_extractor.parameters():
          parameters.to(self.device)
        self.mode = 'with_desc'
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
        
        self.regressor = nn.Sequential(nn.Linear(512 + 6, 512),
                                    #    nn.LayerNorm(2048),
                                       nn.ReLU(),
                                       nn.Linear(512, 256),
                                    #    nn.LayerNorm(512),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Linear(256, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 5))
        
        
        # Initialisation des paramètres
        for block in [self.repr_enricher, 
                      self.regressor]:
            for layer in block:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                for parameters in layer.parameters():
                  parameters.to(self.device)

    def rdy_output(self, output, n_to_fill, cls_pos=None):
        """Extracts the last true positions from LLM output."""
        output = output.last_hidden_state
        maxlen = output.size()[1]
        if cls_pos is not None:
            return output[:, cls_pos]
        return torch.stack([output[i, maxlen - n_missing - 1] for i, n_missing in enumerate(n_to_fill)])
    
    def forward(self, x, desc, n_to_fill):
        # cls is always positioned in a consistent way across examples.
        cls_pos = desc[0].index(self.cls_id)
        # compute features
        text_features = self.feature_extractor(torch.tensor(desc,
                                                            dtype=torch.int,
                                                            device=self.device))

        # prepare features for regression task
        text_features = self.rdy_output(text_features, 
                                        n_to_fill, 
                                        cls_pos=cls_pos).to(self.device)

        # enrich representations
        regression_emb = self.repr_enricher(torch.cat((text_features, x), dim=1))
        output = self.regressor(torch.cat((regression_emb, x), dim=1)) # skip connection
        return output

class CustomMLP(nn.Module):
    def __init__(self, input_dim: int, 
                 layer_dims: List[int], 
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 use_batchnorm: bool = False,
                 use_layernorm: bool = False,
                 dropout: float = 0.1):
        """
        Generic MLP module allowing custom architecture.
        """
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in layer_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(dim))
            layers.append(activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        self.model = nn.Sequential(*layers)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        return self.model(x)

class AttentiveEmbedding(nn.Module):
    def __init__(self, KEY_SIZE, EMB_SIZE, n_head=1, attn_dropout=0.1):
        super().__init__()
        self.attn_layer = nn.MultiheadAttention(EMB_SIZE, kdim=KEY_SIZE, vdim=EMB_SIZE,
                                                num_heads=n_head,
                                                dropout=attn_dropout, batch_first=True)
        self.layernorm = nn.LayerNorm(EMB_SIZE)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key):
        attn_output, _ = self.attn_layer(query, key, query)
        attn_output = self.dropout(attn_output)
        return self.layernorm(query + attn_output)

class AddandNorm(nn.Module):
    def __init__(self, normalizer):
        super().__init__()
        self.normalizer = normalizer
    
    def forward(self, x, to_add):
        return self.normalizer(x + to_add)

class CustomizableFNVModel(BaseModel):
    def __init__(self, llm, device,
                 repr_layers: List[int],
                 regressor_layers: List[int],
                 activation_fn: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 use_batchnorm: bool = False,
                 use_layernorm: bool = False,
                 dropout: float = 0.1,
                 cls_id: int = None,
                 separate_mlp: bool = True,
                 repr_enricher: bool = True,
                 use_attn: bool = True, 
                 attn_dropout: float = None,
                 nheads: int = None,
                 use_skip: bool = False,
                 add_and_norm: bool = False,
                 add_and_norm_normalizer: Callable[[torch.Tensor], torch.Tensor] = None,
                 ):
        """
        Configurable model integrating an LLM and MLP-based regressors.
        """
        super().__init__()
        self.device = device
        self.feature_extractor = llm.to(self.device)
        self.cls_id = cls_id
        self.mode = 'with_desc'
        for param in self.feature_extractor.parameters():
            param.to(self.device)
        
        # Enrich representation or not 
        self.use_repr_enricher = repr_enricher

        # Use a custom attention block or not
        self.use_attn = use_attn

        # Wether to separate energy and protein value regressors
        self.separate_mlp = separate_mlp

        # Wether to use skip connections when possible
        self.use_skip = use_skip

        # When using skip connections, wether to add and normalize or not
        self.add_and_norm = add_and_norm
        self.add_and_norm_normalizer = add_and_norm_normalizer if add_and_norm_normalizer is not None else nn.LayerNorm

        EMB_SIZE = 768
        
        if self.use_skip and self.add_and_norm:
            self.clean_skip = AddandNorm(self.add_and_norm_normalizer(EMB_SIZE))

        if self.use_repr_enricher:
            repr_layers += [EMB_SIZE] # We want to output the same dimension
            # Representation Enricher
            self.repr_enricher = CustomMLP(
                input_dim=6 + EMB_SIZE,
                layer_dims=repr_layers,
                activation_fn=activation_fn,
                use_batchnorm=use_batchnorm,
                use_layernorm=use_layernorm,
                dropout=dropout
            )
        
        if self.use_attn:
            attn_dropout = 0.1 if attn_dropout is None else attn_dropout
            nheads = 2 if nheads is None else nheads
            self.improve_repr = AttentiveEmbedding(6, 
                                                  EMB_SIZE, 
                                                  n_head=nheads,
                                                  attn_dropout=attn_dropout)
        if self.separate_mlp:
            # Protein Regressor
            self.prot_regressor = CustomMLP(
                input_dim=EMB_SIZE + 6,
                layer_dims=regressor_layers + [3],
                activation_fn=activation_fn,
                use_batchnorm=use_batchnorm,
                use_layernorm=use_layernorm,
                dropout=dropout
            )
            
            # Energy Regressor
            self.en_regressor = CustomMLP(
                input_dim=EMB_SIZE + 6,
                layer_dims=regressor_layers + [2],
                activation_fn=activation_fn,
                use_batchnorm=use_batchnorm,
                use_layernorm=use_layernorm,
                dropout=dropout
            )
        else:
            self.regressor = CustomMLP(
                input_dim=EMB_SIZE + 6,
                layer_dims=regressor_layers + [5],
                activation_fn=activation_fn,
                use_batchnorm=use_batchnorm,
                use_layernorm=use_layernorm,
                dropout=dropout
            )
    
    def rdy_output(self, output, n_to_fill, cls_pos=None):
        """Extracts the last true positions from LLM output."""
        output = output.last_hidden_state
        maxlen = output.size()[1]
        if cls_pos is not None:
            return output[:, cls_pos]
        return torch.stack([output[i, maxlen - n_missing - 1] for i, n_missing in enumerate(n_to_fill)])
    
    def forward(self, x, desc, n_to_fill):
        # cls is always positioned in a consistent way across examples.
        cls_pos = desc[0].index(self.cls_id)
        # print(desc)
        # print(self.cls_id)
        # print(cls_pos)
        # print(torch.tensor(desc, dtype=torch.int, device=self.device).size())
        text_features = self.feature_extractor(torch.tensor(desc, dtype=torch.int, device=self.device))
        text_features = self.rdy_output(text_features, n_to_fill, cls_pos=cls_pos).to(self.device)
        
        if self.use_attn:
            if self.use_skip and self.add_and_norm:
                text_features = self.clean_skip(self.improve_repr(text_features, x),
                                                text_features)
            elif self.use_skip and not self.add_and_norm:
                text_features = self.improve_repr(text_features, x) + text_features
            else:
                text_features = self.improve_repr(text_features, x)
        
        if self.use_repr_enricher:
            if self.use_skip:
                text_features = self.clean_skip(self.repr_enricher(torch.cat((text_features, x), dim=1)),
                                                text_features)
            elif self.use_skip and not self.add_and_norm:
                text_features = self.repr_enricher(torch.cat((text_features, x), dim=1)) + text_features
            else:
                text_features = self.repr_enricher(torch.cat((text_features, x), dim=1))
        
        if self.separate_mlp:
            prot_output = self.prot_regressor(torch.cat((text_features, x), dim=1))
            en_output = self.en_regressor(torch.cat((text_features, x), dim=1))
            output = torch.cat((en_output, prot_output), dim=1)
        else:
            output = self.regressor(torch.cat((text_features, x), dim=1))    
        return output

if __name__ == "__main__":
    model = ModeleSansDescription()
    print(model.history)
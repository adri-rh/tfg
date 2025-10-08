#Librerías principales: PyTorch, PyTorch Geometric y Lightning
import os
import sys
import torch
import torch_geometric.datasets as datasets
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torch_geometric.nn import GCNConv, GATConv, GraphConv
import torch_geometric.nn as geom_nn

#Ruta donde se van a guardar los checkpoints del modelo
CHECKPOINT_PATH = "./checkpoints"

#Se selecciona la GPU si está disponible. En caso contrario, la CPU
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#Diccionario para poder elegir el tipo de capa GNN
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

#Hiperparámetros globales
GLOBAL_batch_size = 100
GLOBAL_max_epochs = 50

#La clase GNNModel define por nodo una red de tipo GNN genérica
class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Número de features de entrada por nodo
            c_hidden - Tamaño de las capas ocultas
            c_out - Tamaño de la salida (por ejemplo, número clases)
            num_layers - Número de capas GNN
            layer_name - Tipo de capa (GCN, GAT, GraphConv)
            dp_rate - Tasa de dropout
            kwargs - Parámetros adicionales (por ejemplo, número de 'heads' en GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden

        #Se crean las capas ocultas
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        #La última capa (sin ReLU ni Dropout)
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Características por nodo
            edge_index - Lista con las aristas del grafo en formato PyTorch Geometric
        """
        for l in self.layers:
            #Las capas GNN (MessagePassing) requieren edge_index
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x
    
#La clase GraphGNNModel pasa de nivel nodo a nivel grafo
class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimensión de las características de entrada  
            c_hidden - Dimensión de las características ocultas  
            c_out - Dimensión de las características de salida (normalmente el número de clases)  
            dp_rate_linear - Tasa de dropout antes de la capa lineal (generalmente más alta que dentro de la GNN)  
            kwargs - Parámetros adicionales
        """
        #Modelo GNN para clasificación a nivel de grafo. Combina la GNN de nodos con una capa lineal final.
        super().__init__()

        #Modelo GNN de nodos
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden, # Not our prediction output yet!
                            **kwargs)
        
        #Capa lineal de salida (predicción de clases)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        """
        Inputs:
            x - Características por nodo
            edge_index - Lista con las aristas del grafo en formato PyTorch Geometric
            batch_idx - Identifica a que grafo pertenece cada nodo
        """
        #Propagación de la GNN
        x = self.GNN(x, edge_index)

        #Pooling global: convierte los embeddings de nodos a un vector por grafo
        x = geom_nn.global_mean_pool(x, batch_idx)

        #Capa lineal final
        x = self.head(x)
        return x


#La clase GraphLevelGNN integra el modelo con PyTorch Geometric
class GraphLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()

        #Guarda los hiperparámetros
        self.save_hyperparameters()

        #Modelo base
        self.model = GraphGNNModel(**model_kwargs)

        #Definición de la función de pérdida
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        #Extracción de los datos del batch
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch

        #Paso hacia delante
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)

        #Predicciones y cálculo de pérdida
        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        #Optimizador AdamW con un scheduler para reducir el LR si no mejora
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1),
                "monitor": "train_loss",
                "frequency": 1,
            },
        }

    #Fase de entrenamiento
    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss, prog_bar=True, batch_size=GLOBAL_batch_size)
        self.log('train_acc', acc, prog_bar=True, batch_size = GLOBAL_batch_size)

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True, on_step=True)
        return loss

    #Fase de validación
    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc, prog_bar=True, batch_size=GLOBAL_batch_size)

    #Fase de test
    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc, prog_bar=True, batch_size=GLOBAL_batch_size)

#Entrenamiento del modelo con el dataset MUTAG
if __name__ == '__main__':
    DATASET_PATH = './data'
    #Carga del dataset MUTAG
    tu_dataset = datasets.TUDataset(root=DATASET_PATH, name="MUTAG")

    #Se fija una semilla para la generación de números aleatorios
    torch.manual_seed(42)
    tu_dataset.shuffle()

    #División del dataset en entrenamiento y test
    MUTAG_train_dataset = tu_dataset[:150]
    MUTAG_test_dataset = tu_dataset[150:]

    #DataLoaders (para cargar grafos por lotes)
    MUTAG_graph_train_loader = DataLoader(MUTAG_train_dataset, batch_size=1, shuffle=True)
    MUTAG_graph_val_loader = DataLoader(MUTAG_test_dataset, batch_size=1)  # Additional loader if you want to change to a larger dataset
    MUTAG_graph_test_loader = DataLoader(MUTAG_test_dataset, batch_size=1)
    
    #Número de características y clases del dataset
    num_node_features = tu_dataset.num_node_features
    num_classes = tu_dataset.num_classes
    
    #Inicialización del modelo
    pl_model = GraphLevelGNN(
        c_in=num_node_features, 
        c_hidden=64, 
        c_out=num_classes,
        layer_name="GCN", 
        num_layers=2
    )
    
    #Nombre y ruta para guardar el modelo
    model_name = "GraphConv" 
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)

    #Entrenamiento con PyTorch Geometric
    trainer = pl.Trainer(default_root_dir=root_dir,
                              callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    EarlyStopping('val_acc')],
                              accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                              devices=1,
                              max_epochs=GLOBAL_max_epochs,
                              enable_progress_bar=True,
                              enable_model_summary=True,
                              num_sanity_val_steps=5,
                              logger=False)
    
    #Entrenamiento y validación
    trainer.fit(pl_model, train_dataloaders=MUTAG_graph_train_loader, val_dataloaders=MUTAG_graph_val_loader)
    
    #Resultado final del test
    test_result = trainer.test(pl_model, dataloaders=MUTAG_graph_test_loader)
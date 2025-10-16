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
import torch_geometric.nn as geom_nn
from pytorch_lightning.loggers import WandbLogger
import wandb

#Ruta donde se van a guardar los checkpoints del modelo
CHECKPOINT_PATH = "./checkpoints"

#Se selecciona la GPU si está disponible
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

#Diccionario para elegir el tipo de capa GNN
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

#Hiperparámetros globales
GLOBAL_batch_size = 8
GLOBAL_max_epochs = 50

#Definición de clases del modelo
class GNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GAT", dp_rate=0.1, heads=4, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels = c_in

        #Capas intermedias con múltiples cabezas de atención
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels, out_channels=c_hidden, heads=heads, concat=True, **kwargs),
                nn.ELU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden * heads

        #Capa final con una sola cabeza (para reducir dimensionalidad)
        layers += [gnn_layer(in_channels=in_channels, out_channels=c_out, heads=1, concat=True, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x

#La clase GraphGNNModel pasa de nivel nodo a nivel grafo
class GraphGNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        super().__init__()
        self.GNN = GNNModel(c_in=c_in, c_hidden=c_hidden, c_out=c_hidden, **kwargs)
        self.head = nn.Sequential(nn.Dropout(dp_rate_linear), nn.Linear(c_hidden, c_out))

    def forward(self, x, edge_index, batch_idx):
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)
        x = self.head(x)
        return x

#La clase GraphLevelGNN integra el modelo con PyTorch Geometric
class GraphLevelGNN(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss() if self.hparams.c_out == 1 else nn.CrossEntropyLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(dim=-1)
        if self.hparams.c_out == 1:
            preds = (x > 0).float()
            data.y = data.y.float()
        else:
            preds = x.argmax(dim=-1)
        loss = self.loss_module(x, data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1),
                "monitor": "train_loss",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx):
        loss, acc = self.forward(batch, mode="train")
        self.log('train_loss', loss, prog_bar=True, batch_size=GLOBAL_batch_size)
        self.log('train_acc', acc, prog_bar=True, batch_size=GLOBAL_batch_size)
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc, prog_bar=True, batch_size=GLOBAL_batch_size)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc, prog_bar=True, batch_size=GLOBAL_batch_size)


#Entrenamiento con MUTAG
if __name__ == '__main__':
    DATASET_PATH = './data'
    tu_dataset = datasets.TUDataset(root=DATASET_PATH, name="MUTAG")
    #torch.manual_seed(42)
    tu_dataset.shuffle()
    MUTAG_train_dataset = tu_dataset[:150]
    MUTAG_test_dataset = tu_dataset[150:]
    MUTAG_graph_train_loader = DataLoader(MUTAG_train_dataset, batch_size=1, shuffle=True)
    MUTAG_graph_val_loader = DataLoader(MUTAG_test_dataset, batch_size=1)
    MUTAG_graph_test_loader = DataLoader(MUTAG_test_dataset, batch_size=1)
    num_node_features = tu_dataset.num_node_features
    num_classes = tu_dataset.num_classes

    #Modelo GAT
    pl_model = GraphLevelGNN(
        c_in=num_node_features,
        c_hidden=64,
        c_out=num_classes,
        layer_name="GAT",
        num_layers=4,
        dp_rate=0.4,
        heads=6
    )

    model_name = "GAT"
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)

    #Inicializar wandb
    wandb_logger = WandbLogger(
        project="tfg",
        name="GAT_MUTAG_4",
        log_model=True
    )

    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    EarlyStopping('val_acc')],
                         accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=GLOBAL_max_epochs,
                         enable_progress_bar=True,
                         logger=wandb_logger)

    trainer.fit(pl_model, train_dataloaders=MUTAG_graph_train_loader, val_dataloaders=MUTAG_graph_val_loader)
    test_result = trainer.test(pl_model, dataloaders=MUTAG_graph_test_loader)
    wandb.finish()
#Librerías necesarias
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as geom_nn
import torch_geometric.transforms as T
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
from pytorch_lightning.loggers import WandbLogger
import wandb

#Configuración global
CHECKPOINT_PATH = "./checkpoints"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f"Usando dispositivo: {device}")

GLOBAL_BATCH_SIZE = 100
GLOBAL_MAX_EPOCHS = 50

#Modelo GNN base
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

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

        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels, out_channels, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        layers += [gnn_layer(in_channels, c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Características por nodo
            edge_index - Lista con las aristas del grafo en formato PyTorch Geometric
        """
        for l in self.layers:
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
        super().__init__()
        self.GNN = GNNModel(c_in, c_hidden, c_hidden, **kwargs)
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
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)
        return self.head(x)

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
        preds = (x > 0).float() if self.hparams.c_out == 1 else x.argmax(dim=-1)
        loss = self.loss_module(x, data.y.float() if self.hparams.c_out == 1 else data.y)
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}

    #Fase de entrenamiento
    def training_step(self, batch, _):
        loss, acc = self.forward(batch, "train")
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    #Fase de validación
    def validation_step(self, batch, _):
        _, acc = self.forward(batch, "val")
        self.log('val_acc', acc, prog_bar=True)

    #Fase de test
    def test_step(self, batch, _):
        _, acc = self.forward(batch, "test")
        self.log('test_acc', acc, prog_bar=True)

#Dataset JAAD -> Crea el dataset de grafos a partir de los .csv
class JAAD(InMemoryDataset):
    @property
    def processed_file_names(self):
        return ['data_jaad.pt', 'data_jaad_test.pt']

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None, mode='train'):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.mode = mode
        if self.mode == 'train':
            self.load(self.processed_paths[0])
        elif self.mode == 'test':
            self.load(self.processed_paths[1])
        else:
            raise ValueError('Modo no válido (usa train o test)')

    def process(self):
        def create_graphs(csv_path, desc):
            df = pd.read_csv(csv_path)
            n_rows = len(df.index)
            n_feats = len([df[e].unique().size for e in df.drop(columns=['ped_id', 'cross'])])

            #Construcción de características por nodo
            x = np.vstack([
                np.pad(np.zeros(n_rows)[np.newaxis].T, [(0, 0), (0, 19)], 'constant'),
                np.pad(pd.get_dummies(df['attention']).to_numpy(dtype=float), [(0, 0), (1, 19 - df['attention'].nunique())], 'constant'),
                np.pad(pd.get_dummies(df['orientation']).to_numpy(dtype=float), [(0, 0), (3, 17 - df['orientation'].nunique())], 'constant'),
                np.pad(pd.get_dummies(df['proximity']).to_numpy(dtype=float), [(0, 0), (7, 13 - df['proximity'].nunique())], 'constant'),
                np.pad(pd.get_dummies(df['distance']).to_numpy(dtype=float), [(0, 0), (10, 10 - df['distance'].nunique())], 'constant'),
                np.pad(pd.get_dummies(df['action']).to_numpy(dtype=float), [(0, 0), (15, 5 - df['action'].nunique())], 'constant')
            ])

            data_list = []
            for i in tqdm(range(n_rows), desc=desc):
                x_ = np.empty((0, 20))
                x_ = np.vstack([x_, x[i]])
                for j in range(n_feats):
                    x_ = np.vstack([x_, x[n_rows * (j + 1) + i]])

                #Grafo de Angie
                #edges_ = np.array([[0, 0, 0, 0, 0],
                #                    [1, 2, 3, 4, 5]], dtype=int)

                #Grafo de Adrián
                #edges_ = np.array([[0, 0, 0, 0, 4],
                #                    [1, 2, 3, 4, 5]], dtype=int)

                #Grafo propuesto en la reunión
                edges_ = np.array([[0, 0, 0, 0, 0, 4],
                                    [1, 2, 3, 4, 5, 5]], dtype=int)

                label = torch.tensor([df['cross'].factorize(['noCrossRoad', 'CrossRoad'])[0][i]], dtype=torch.long)
                graph = Data(x=torch.tensor(x_).float(),
                             edge_index=torch.tensor(edges_, dtype=torch.long),
                             y=label)
                data_list.append(graph)
            return data_list

        data_list = create_graphs('datasets/dataJAAD.csv', "Procesando JAAD (train)")
        self.save(data_list, self.processed_paths[0])
        data_list_test = create_graphs('datasets/JAAD_TEST.csv', "Procesando JAAD (test)")
        self.save(data_list_test, self.processed_paths[1])


#Entrenamiento del modelo con el dataset JAAD
if __name__ == "__main__":
    pl.seed_everything(42)

    dts = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='train')
    dts_test = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='test')

    train_loader = DataLoader(dts[:13000], batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=8)
    val_loader = DataLoader(dts[13000:14000], batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=8)
    test_loader = DataLoader(dts_test, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=8)

    model = GraphLevelGNN(c_in=20, c_out=1, c_hidden=256,
                          dp_rate_linear=0.5, dp_rate=0.0,
                          num_layers=3, layer_name="GraphConv")

    wandb_logger = WandbLogger(
        project="tfg",
        name="GraphConv_JAAD_grafo_Combinación",
        log_model=True
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "GraphLevelJAAD"),
        callbacks=[ModelCheckpoint(save_weights_only=True, monitor="val_acc", mode="max"),
                   EarlyStopping('val_acc')],
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=GLOBAL_MAX_EPOCHS,
        enable_progress_bar=True,
        logger=wandb_logger
    )

    trainer.fit(model, train_loader, val_loader)
    best_model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    print("\nEvaluación final del modelo JAAD:")
    trainer.test(best_model, dataloaders=test_loader, verbose=True)
    wandb.finish()

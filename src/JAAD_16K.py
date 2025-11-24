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
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    #Fase de entrenamiento
    def training_step(self, batch, _):
        loss, acc = self.forward(batch, "train")
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    #Fase de validación
    def validation_step(self, batch, _):
        loss, acc = self.forward(batch, "val")
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

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
            #Ignorar columnas no necesarias
            df = df.drop(columns=['video', 'frame', 'person'])
            n_rows = len(df.index)
            n_feats = 6

            #Construcción de características por nodo
            def pad_feature(arr, left, right, total=24):
                arr_padded = np.pad(arr, [(0, 0), (left, max(0, right))], 'constant')
                #Forzar tamaño uniforme (rellena con ceros si falta)
                if arr_padded.shape[1] < total:
                    arr_padded = np.pad(arr_padded, [(0, 0), (0, total - arr_padded.shape[1])], 'constant')
                elif arr_padded.shape[1] > total:
                    arr_padded = arr_padded[:, :total]
                return arr_padded

            #Construcción de nodos para el dataset numérico
            x = np.vstack([
            pad_feature(np.zeros(n_rows)[np.newaxis].T, 0, 19),
            pad_feature(pd.get_dummies(df['attention']).to_numpy(dtype=float), 1, 19 - df['attention'].nunique()),
            pad_feature(pd.get_dummies(df['orientation']).to_numpy(dtype=float), 3, 17 - df['orientation'].nunique()),
            pad_feature(pd.get_dummies(df['proximity']).to_numpy(dtype=float), 7, 13 - df['proximity'].nunique()),
            pad_feature(df[['distance']].to_numpy(dtype=float), 10, 9),
            pad_feature(pd.get_dummies(df['action']).to_numpy(dtype=float), 15, 5 - df['action'].nunique()),
            pad_feature(pd.get_dummies(df['zebra_cross']).to_numpy(dtype=float), 18, 2 - df['zebra_cross'].nunique())
            ])

            """data_list = []
            for i in tqdm(range(n_rows), desc=desc):
                x_ = np.empty((0, 24))
                x_ = np.vstack([x_, x[i]])
                for j in range(n_feats):
                    x_ = np.vstack([x_, x[n_rows * (j + 1) + i]])

            #Grafo de Angie
            #edges_ = np.array([[0, 0, 0, 0, 0],
            #                    [1, 2, 3, 4, 5]], dtype=int)

            #Grafo de Angie (lingüístico)
            #edges_ = np.array([[0, 0, 0, 0, 0, 0],
            #                    [1, 2, 3, 4, 5, 6]], dtype=int)

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
            return data_list"""

            #Grafo espaciotemporal
            data_list = []
            window_size = 3 #Número de frames por grafo
            step = 1

            for start in tqdm(range(0, n_rows - window_size, step), desc=desc):
                end = start + window_size
                #Agrupar los frames consecutivos
                frames_window = df.iloc[start:end]
                x_window = []
                edge_index = []

                #Grafo espaciotemporal completo
                """for i, row_idx in enumerate(range(start, end)):
                    x_frame = np.empty((0, 24))
                    x_frame = np.vstack([x_frame, x[row_idx]])
                    for j in range(n_feats):
                        x_frame = np.vstack([x_frame, x[n_rows * (j + 1) + row_idx]])
                    x_window.append(x_frame)"""

                #Grafo de Angie
                #spatial_edges = np.array([[0, 0, 0, 0, 0],
                #                        [1, 2, 3, 4, 5]], dtype=int) + i * (n_feats + 1)
                #edge_index.append(spatial_edges)

                #Grafo de Adrián
                #spatial_edges = np.array([[0, 0, 0, 0, 4],
                #                        [1, 2, 3, 4, 5]], dtype=int) + i * (n_feats + 1)
                #edge_index.append(spatial_edges)

                #Grafo propuesto en la reunión
                #spatial_edges = np.array([[0, 0, 0, 0, 0, 4],
                #                        [1, 2, 3, 4, 5, 5]], dtype=int) + i * (n_feats + 1)
                #edge_index.append(spatial_edges)

                    #Conexión temporal entre nodos pedestrian de frames consecutivos
                    #"""if i > 0:
                    #    temporal_edge = np.array([[0 + (i - 1) * (n_feats + 1)],
                    #                            [0 + i * (n_feats + 1)]], dtype=int)
                    #    edge_index.append(temporal_edge)"""

                    #Conexiones temporales entre todos los nodos equivalentes de frames consecutivos
                    #"""if i > 0:
                    #    for node_id in range(n_feats + 1):
                    #        prev_node = node_id + (i - 1) * (n_feats + 1)
                    #        curr_node = node_id + i * (n_feats + 1)
                    #        temporal_edge = np.array([[prev_node], [curr_node]], dtype=int)
                    #        edge_index.append(temporal_edge)"""
                
                #Promedio para el sliding windows
                x_window = []
                for j in range(n_feats + 1):
                    idxs = [n_rows * j + i for i in range(start, end)]
                    x_window.append(x[idxs].mean(axis=0))

                #Grafo de Angie
                #edges_ = np.array([[0, 0, 0, 0, 0],
                #                    [1, 2, 3, 4, 5]], dtype=int)

                #Grafo de Adrián
                #edges_ = np.array([[0, 0, 0, 0, 4],
                #                    [1, 2, 3, 4, 5]], dtype=int)

                #Grafo propuesto en la reunión
                edges_ = np.array([[0, 0, 0, 0, 0, 4],
                                    [1, 2, 3, 4, 5, 5]], dtype=int)


                #Unir todo
                x_combined = np.vstack(x_window)
                #edges_combined = np.hstack(edge_index) #Combinar frames y edges
                edges_combined = edges_ #Combinar solo edges
                encoded_labels = frames_window['cross'].map({'not-crossing': 0, 'crossing': 1, 'noCrossRoad': 0, 'CrossRoad': 1})
                label = torch.tensor([encoded_labels.mode()[0]], dtype=torch.long)

                graph = Data(
                    x=torch.tensor(x_combined).float(),
                    edge_index=torch.tensor(edges_combined, dtype=torch.long),
                    y=label
                )
                data_list.append(graph)

            return data_list

        data_list = create_graphs('datasets/JAAD_16K_TRAIN.csv', "Procesando JAAD (train)")
        self.save(data_list, self.processed_paths[0])
        data_list_test = create_graphs('datasets/TEST_JAAD_ALL.csv', "Procesando JAAD (test)")
        self.save(data_list_test, self.processed_paths[1])


#Entrenamiento del modelo con el dataset JAAD
if __name__ == "__main__":
    pl.seed_everything(42)
    #Determinismo
    #torch.use_deterministic_algorithms(True)
    #os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    for f in ["data/processed/data_jaad.pt", "data/processed/data_jaad_test.pt"]:
        if os.path.exists(f):
            os.remove(f)
            print("Borrado:", f)

    dts = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='train')
    dts_test = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='test')

    #División dinámica (80% train / 20% val)
    train_size = int(0.8 * len(dts))
    val_size = len(dts) - train_size
    print(f"Dataset total: {len(dts)} | Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(dts[:train_size], batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(dts[train_size:], batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(dts_test, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=4)

    model = GraphLevelGNN(c_in=24, c_out=1, c_hidden=256,
                          dp_rate_linear=0.5, dp_rate=0.0,
                          num_layers=3, layer_name="GraphConv")

    wandb_logger = WandbLogger(
        project="tfg",
        name="GraphConv_JAAD_16K_SlidingWindows_Combinación",
        log_model=True
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "GraphLevelJAAD"),
        callbacks=[ModelCheckpoint(save_weights_only=True, monitor="val_acc", mode="max"), 
                   EarlyStopping(monitor="val_loss", patience=3, mode="min")],
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

#Probar las 30 semillas
"""if __name__ == "__main__":
    seeds = list(range(30))  #30 semillas distintas
    results = []

    for seed in seeds:
        print(f"\n🔹 Ejecutando experimento con semilla {seed}")
        pl.seed_everything(seed)

        dts = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='train')
        dts_test = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='test')

        #División dinámica (80% train / 20% val)
        train_size = int(0.8 * len(dts))
        val_size = len(dts) - train_size
        print(f"Dataset total: {len(dts)} | Train: {train_size} | Val: {val_size}")

        train_loader = DataLoader(dts[:train_size], batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader = DataLoader(dts[train_size:], batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader = DataLoader(dts_test, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=4)

        model = GraphLevelGNN(
            c_in=24, c_out=1, c_hidden=256,
            dp_rate_linear=0.5, dp_rate=0.0,
            num_layers=3, layer_name="GraphConv"
        )

        wandb_logger = WandbLogger(
            project="tfg",
            name=f"GraphConv_JAAD_robustez_seed_{seed}",
            log_model=False
        )

        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, f"seed_{seed}"),
            callbacks=[
                ModelCheckpoint(save_weights_only=True, monitor="val_acc", mode="max"),
                EarlyStopping(monitor="val_loss", patience=3, mode="min")
            ],
            accelerator="gpu" if str(device).startswith("cuda") else "cpu",
            devices=1,
            max_epochs=GLOBAL_MAX_EPOCHS,
            enable_progress_bar=True,
            logger=wandb_logger
        )

        trainer.fit(model, train_loader, val_loader)

        best_model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        test_result = trainer.test(best_model, dataloaders=test_loader, verbose=False)
        test_acc = test_result[0]["test_acc"]
        print(f"✅ Seed {seed} -> Test Accuracy: {test_acc:.4f}")
        results.append({"seed": seed, "test_acc": test_acc})

        wandb.log({"test_acc": test_acc})
        wandb.finish()"""
"""
El siguiente script está preparado para generar datasets de grafos (espaciotemporales, ventanas deslizantes
y variantes especiales), con modelos GNN (GCN, GAT y GraphConv), utilizando PyTorch Lightning.
Abajo aparece toda la configuración mediante variables globales (GRAPH_TYPE, TEMPORAL_TYPE, DATASET_NAME, etc).
Actualmente, se incluyen dos mains:
 1. Main estándar (se encuentra activo).
 2. Main para probar 30 semillas de golpe (se encuentra comentado).
"""

#Imports
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

#Configuración global (Las variables son modificables para probar los distintos casos)

#Modo multipeatón (True = grafo por escena (multi-pedestrian), False = grafo por peatón/temporal (single-pedestrian))
MULTI_PEDESTRIAN = True

#Opciones de grafos y opciones temporales

#Tipo de grafo espacial
#0 = Angie
#1 = Adrián
#2 = Combinación del grafo 0 y grafo 1
GRAPH_TYPE = 0

#Tipo de dependencias temporales
#0 = Sin conexiones temporales
#1 = Conexión peatón entre frames
#2 = Conexión completa entre nodos
#3 = Ventanas deslizantes
#4 = Ventanas de ESCENAS multi-peatón
TEMPORAL_TYPE = 4

#Parámetros de las ventanas deslizantes (exclusivo de TEMPORAL_TYPE = 3)
WINDOW_SIZE = 3
WINDOW_STEP = 1

#Opciones de datasets y opciones de ejecución
DATA_FOLDER = "datasets"

#4 datasets disponibles
DATASETS = {
    "JAAD_14K": {
        "train": os.path.join(DATA_FOLDER, "JAAD_14K_TRAIN.csv"),
        "test" : os.path.join(DATA_FOLDER, "TEST_JAAD_ALL.csv")
    },
    "JAAD_16K": {
        "train": os.path.join(DATA_FOLDER, "JAAD_16K_TRAIN.csv"),
        "test" : os.path.join(DATA_FOLDER, "TEST_JAAD_ALL.csv")
    },
    "JAAD_18K": {
        "train": os.path.join(DATA_FOLDER, "JAAD_18K_TRAIN.csv"),
        "test" : os.path.join(DATA_FOLDER, "TEST_JAAD_ALL.csv")
    },
    "LING_14K": {
        "train": os.path.join(DATA_FOLDER, "LINGUISTIC_JAAD_TRAIN_14K.csv"),
        "test" : os.path.join(DATA_FOLDER, "LINGUISTIC_TEST_JAAD_ALL.csv")
    }
}

#Selección de dataset para ejecutar
DATASET_NAME = "JAAD_14K"

#Entrenamiento
GLOBAL_BATCH_SIZE = 100
GLOBAL_MAX_EPOCHS = 50
CHECKPOINT_PATH = "./checkpoints"

#Logger WandB (Activar = True, desactivar = False)
USE_WANDB = True
WANDB_PROJECT = "tfg"

#Modelos e hiperparámetros
MODEL_NAME = "GraphConv"   #"GCN", "GAT", "GraphConv"
C_IN = 24
C_HIDDEN = 128
C_OUT = 1
DP_RATE_LINEAR = 0.5
DP_RATE = 0.2
NUM_LAYERS = 2

#Construcción de los grafos
def build_spatial_edges(graph_type):
    if graph_type == 0:
        return np.array([[0, 0, 0, 0, 0],
                         [1, 2, 3, 4, 5]], dtype=int)
    elif graph_type == 1:
        return np.array([[0, 0, 0, 0, 4],
                         [1, 2, 3, 4, 5]], dtype=int)
    elif graph_type == 2:
        return np.array([[0, 0, 0, 0, 0, 4],
                         [1, 2, 3, 4, 5, 5]], dtype=int)
    else:
        raise ValueError("graph_type inválido")


def build_temporal_edges_between_frames(temporal_type, prev_offset, curr_offset, n_feats):
    if temporal_type == 0 or temporal_type == 3:
        return []
    edges = []
    if temporal_type == 1:
        edges.append(np.array([[prev_offset + 0], [curr_offset + 0]]))
    elif temporal_type == 2:
        for node in range(n_feats + 1):
            edges.append(np.array([[prev_offset + node], [curr_offset + node]]))
    return edges

#Modelos
gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

class GNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]
        layers = []
        in_channels = c_in

        for _ in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels, c_hidden, **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden

        layers += [gnn_layer(in_channels, c_out, **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for l in self.layers:
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x


class GraphGNNModel(nn.Module):
    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        super().__init__()
        self.GNN = GNNModel(c_in, c_hidden, c_hidden, **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx)
        return self.head(x)


class GraphLevelGNN(pl.LightningModule):
    def __init__(self, **model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = GraphGNNModel(**model_kwargs)
        self.loss_module = nn.BCEWithLogitsLoss()

    def forward(self, data, mode="train"):
        x, edge_index, batch_idx = data.x, data.edge_index, data.batch
        x = self.model(x, edge_index, batch_idx)
        x = x.squeeze(-1)
        preds = (x > 0).float()
        loss = self.loss_module(x, data.y.float())
        acc = (preds == data.y).sum().float() / preds.shape[0]
        return loss, acc

    def training_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, acc = self.forward(batch)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, _):
        _, acc = self.forward(batch)
        self.log('test_acc', acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

#Dataset JAAD
class JAAD(InMemoryDataset):
    @property
    def processed_file_names(self):
        return ['data_jaad.pt', 'data_jaad_test.pt']

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 mode='train', csv_path_train=None, csv_path_test=None,
                 graph_type=2, temporal_type=3, window_size=3, window_step=1):

        self._csv_train = csv_path_train
        self._csv_test = csv_path_test
        self._graph_type = graph_type
        self._temporal_type = temporal_type
        self._window_size = window_size
        self._window_step = window_step

        super().__init__(root, transform, pre_transform, pre_filter)

        self.mode = mode
        if mode == "train":
            self.load(self.processed_paths[0])
        else:
            self.load(self.processed_paths[1])

    def process(self):

        #Función para crear grafos
        def create_graphs(csv_path, desc):

            df = pd.read_csv(csv_path)
            df['cross'] = df['cross'].astype(str)

            n_rows = len(df)
            n_feats = 6

            #Feature encoding
            def pad_feature(arr, left, right, total=24):
                arr = np.pad(arr, [(0, 0), (left, max(0, right))])
                if arr.shape[1] < total:
                    arr = np.pad(arr, [(0, 0), (0, total - arr.shape[1])])
                return arr[:, :total]

            block_list = [
                pad_feature(np.zeros((n_rows, 1)), 0, 19),
                pad_feature(pd.get_dummies(df.get('attention', 0)).to_numpy(float), 1, 19),
                pad_feature(pd.get_dummies(df.get('orientation', 0)).to_numpy(float), 3, 17),
                pad_feature(pd.get_dummies(df.get('proximity', 0)).to_numpy(float), 7, 13),
                pad_feature(df[['distance']].to_numpy(float), 10, 9),
                pad_feature(pd.get_dummies(df.get('action', 0)).to_numpy(float), 15, 5),
                pad_feature(pd.get_dummies(df.get('zebra_cross', 0)).to_numpy(float), 18, 2),
            ]

            x = np.vstack(block_list)
            spatial_base = build_spatial_edges(self._graph_type)
            data_list = []

            #Feature encoding
            if MULTI_PEDESTRIAN:

                scene_groups = df.groupby(["video", "frame"])

                for (video, frame), scene_df in tqdm(scene_groups, desc=desc):

                    row_idxs = scene_df.index.to_numpy()
                    x_scene = x[row_idxs]
                    num_nodes = x_scene.shape[0]

                    if num_nodes < 2:
                        continue  #Descartar escenas con 1 peatón

                    edges = []
                    for i in range(num_nodes):
                        for j in range(num_nodes):
                            if i != j:
                                edges.append([i, j])

                    edge_index = torch.tensor(edges, dtype=torch.long).T

                    #Etiqueta cross por mayoría
                    """label_val = scene_df['cross'].map(
                        {'not-crossing': 0, 'crossing': 1,
                        'noCrossRoad': 0, 'CrossRoad': 1}
                    ).mode()[0]"""

                    #Etiqueta cross positiva si al menos cruza un peatón
                    label_val = scene_df['cross'].map(
                        {'not-crossing': 0, 'crossing': 1,
                        'noCrossRoad': 0, 'CrossRoad': 1}
                    ).max()

                    graph = Data(
                        x=torch.tensor(x_scene).float(),
                        edge_index=edge_index,
                        y=torch.tensor([label_val], dtype=torch.long)
                    )

                    data_list.append(graph)

                return data_list


            #Eliminar columnas sobrantes si existen
            for col in ['video', 'frame', 'person']:
                if col in df.columns:
                    df = df.drop(columns=[col])

            #Feature encoding
            if self._temporal_type == 3:

                for start in tqdm(range(0, n_rows - self._window_size + 1, self._window_step), desc=desc):
                    end = start + self._window_size
                    frames_window = df.iloc[start:end]

                    x_window = []
                    for f in range(n_feats + 1):
                        idxs = [i + n_rows * f for i in range(start, end)]
                        x_window.append(x[idxs].mean(axis=0))

                    x_combined = np.vstack(x_window)
                    edges_combined = spatial_base.copy()

                    label_val = frames_window['cross'].map(
                        {'not-crossing':0, 'crossing':1, 'noCrossRoad':0, 'CrossRoad':1}
                    ).mode()[0]

                    graph = Data(
                        x=torch.tensor(x_combined).float(),
                        edge_index=torch.tensor(edges_combined),
                        y=torch.tensor([label_val], dtype=torch.long)
                    )
                    data_list.append(graph)

            #Grafos espaciotemporales
            else:

                for start in tqdm(range(0, n_rows - self._window_size + 1, self._window_step), desc=desc):
                    end = start + self._window_size
                    frames_window = df.iloc[start:end]

                    x_blocks = []
                    edges = []

                    for i_frame, frame_idx in enumerate(range(start, end)):
                        xf = np.empty((0, 24))
                        xf = np.vstack([xf, x[frame_idx]])
                        for j in range(n_feats):
                            xf = np.vstack([xf, x[n_rows*(j+1)+frame_idx]])

                        x_blocks.append(xf)

                        offset = i_frame * (n_feats+1)
                        edges.append(spatial_base + offset)

                        if i_frame > 0:
                            prev_o = (i_frame-1)*(n_feats+1)
                            curr_o = i_frame*(n_feats+1)
                            temporal_edges = build_temporal_edges_between_frames(self._temporal_type, prev_o, curr_o, n_feats)
                            edges.extend(temporal_edges)

                    x_combined = np.vstack(x_blocks)
                    edges_combined = np.hstack(edges) if len(edges)>0 else np.zeros((2,0), int)

                    label_val = frames_window['cross'].map(
                        {'not-crossing':0, 'crossing':1, 'noCrossRoad':0, 'CrossRoad':1}
                    ).mode()[0]

                    graph = Data(
                        x=torch.tensor(x_combined).float(),
                        edge_index=torch.tensor(edges_combined),
                        y=torch.tensor([label_val], dtype=torch.long)
                    )
                    data_list.append(graph)

            return data_list
        
        #Función para crear grafos multipeatón temporales
        def create_scene_temporal_graphs(csv_path, desc):
            df = pd.read_csv(csv_path)
            df['cross'] = df['cross'].astype(str)

            scene_groups = df.groupby(["video", "frame"])
            scenes = list(scene_groups)

            data_list = []
            n_feats = 24

            for i in tqdm(range(1, len(scenes) - 1), desc=desc):
                graphs_x = []
                graphs_edges = []

                #Inicializamos offsets
                offset = 0
                prev_offsets = []  #Se guarda el offset de cada escena t-1, t, t+1

                #Escenas t-1, t, t+1
                for t, (_, scene_df) in enumerate(scenes[i-1:i+2]):
                    scene_df = scene_df.reset_index(drop=True)
                    num_nodes = len(scene_df)

                    if num_nodes < 2:
                        break  #Descartar escenas con 1 peatón

                    #Features
                    block_list = [
                        np.zeros((num_nodes, 1)),
                        pd.get_dummies(scene_df.get('attention', 0)).to_numpy(float),
                        pd.get_dummies(scene_df.get('orientation', 0)).to_numpy(float),
                        pd.get_dummies(scene_df.get('proximity', 0)).to_numpy(float),
                        scene_df[['distance']].to_numpy(float),
                        pd.get_dummies(scene_df.get('action', 0)).to_numpy(float),
                        pd.get_dummies(scene_df.get('zebra_cross', 0)).to_numpy(float),
                    ]

                    x_scene = np.hstack(block_list)
                    x_scene = np.pad(x_scene, ((0, 0), (0, n_feats - x_scene.shape[1])))

                    graphs_x.append(x_scene)

                    #Mapeo de peatones a nodos locales
                    person_to_node = {pid: idx for idx, pid in enumerate(scene_df['person'].values)}

                    #Aristas espaciales
                    spatial_edges = []
                    for u in range(num_nodes):
                        for v in range(num_nodes):
                            if u != v:
                                spatial_edges.append([offset + u, offset + v])
                    graphs_edges.append(np.array(spatial_edges).T)

                    #Aristas temporales
                    if t > 0:
                        prev_scene_df = scenes[i-1 + t - 1][1].reset_index(drop=True)
                        prev_person_to_node = {pid: idx for idx, pid in enumerate(prev_scene_df['person'].values)}
                        prev_offset = prev_offsets[t-1]

                        for pid in scene_df['person'].unique():
                            if pid in prev_person_to_node:
                                u = prev_person_to_node[pid]
                                v = person_to_node[pid]
                                graphs_edges.append(
                                    np.array([[prev_offset + u], [offset + v]])
                                )

                    prev_offsets.append(offset)
                    offset += num_nodes

                #Solo seguimos si tenemos las 3 escenas
                if len(graphs_x) < 3:
                    continue

                #Combinamos features y edges
                x_combined = np.vstack(graphs_x)
                edge_index = np.hstack(graphs_edges) if len(graphs_edges) > 0 else np.zeros((2, 0), int)

                #La etiqueta es la escena central
                #Etiqueta cross por mayoría
                """label_val = scenes[i][1]['cross'].map(
                    {'not-crossing': 0, 'crossing': 1, 
                    'noCrossRoad': 0, 'CrossRoad': 1}
                ).mode()[0]"""

                #Etiqueta cross positiva si al menos cruza un peatón
                label_val = scene_df['cross'].map(
                    {'not-crossing': 0, 'crossing': 1,
                    'noCrossRoad': 0, 'CrossRoad': 1}
                ).max()

                graph = Data(
                    x=torch.tensor(x_combined).float(),
                    edge_index=torch.tensor(edge_index, dtype=torch.long),
                    y=torch.tensor([label_val], dtype=torch.long)
                )

                data_list.append(graph)

            return data_list

        #Crear entrenamiento y test
        if MULTI_PEDESTRIAN and self._temporal_type == 4:
            train_list = create_scene_temporal_graphs(self._csv_train, "Procesando JAAD temporal (train)")
        else:
            train_list = create_graphs(self._csv_train, "Procesando JAAD (train)")
        self.save(train_list, self.processed_paths[0])

        if MULTI_PEDESTRIAN and self._temporal_type == 4:
            test_list = create_scene_temporal_graphs(self._csv_test, "Procesando JAAD temporal (test)")
        else:
            test_list = create_graphs(self._csv_test, "Procesando JAAD (test)")
        self.save(test_list, self.processed_paths[1])

#Main principal - Ejecución normal
if __name__ == "__main__":
    pl.seed_everything(42)

    #Borrar los datos procesados
    for f in ["data/processed/data_jaad.pt", "data/processed/data_jaad_test.pt"]:
        if os.path.exists(f):
            os.remove(f)
            print("Borrado:", f)

    csv_train = DATASETS[DATASET_NAME]["train"]
    csv_test  = DATASETS[DATASET_NAME]["test"]

    dts = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='train',
               csv_path_train=csv_train, csv_path_test=csv_test,
               graph_type=GRAPH_TYPE, temporal_type=TEMPORAL_TYPE,
               window_size=WINDOW_SIZE, window_step=WINDOW_STEP)

    dts_test = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='test',
                    csv_path_train=csv_train, csv_path_test=csv_test,
                    graph_type=GRAPH_TYPE, temporal_type=TEMPORAL_TYPE,
                    window_size=WINDOW_SIZE, window_step=WINDOW_STEP)

    train_size = int(0.8 * len(dts))
    val_size = len(dts) - train_size
    print(f"Dataset total: {len(dts)} | Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(dts[:train_size], batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader   = DataLoader(dts[train_size:], batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(dts_test, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=4)

    model = GraphLevelGNN(
        c_in=C_IN, c_out=C_OUT, c_hidden=C_HIDDEN,
        dp_rate_linear=DP_RATE_LINEAR, dp_rate=DP_RATE,
        num_layers=NUM_LAYERS, layer_name=MODEL_NAME
    )

    wandb_logger = WandbLogger(
        project=WANDB_PROJECT,
        name=f"{MODEL_NAME}_{DATASET_NAME}_Grafo {GRAPH_TYPE}_Dependencia temporal {TEMPORAL_TYPE}_Multipeatón {MULTI_PEDESTRIAN}",
        log_model=True
    ) if USE_WANDB else None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    print("\nEvaluación final:")
    trainer.test(best_model, dataloaders=test_loader, verbose=True)

    if USE_WANDB:
        wandb.finish()

#Main secundario - Ejecución de prueba para las 30 semillas
"""
if __name__ == "__main__":

    seeds = list(range(30))
    results = []

    for seed in seeds:
        print(f"\nEjecutando experimento con semilla {seed}")
        pl.seed_everything(seed)

        #Borrar los datos procesados
        for f in ["data/processed/data_jaad.pt", "data/processed/data_jaad_test.pt"]:
            if os.path.exists(f):
                os.remove(f)
                print("Borrado:", f)

        csv_train = DATASETS[DATASET_NAME]["train"]
        csv_test  = DATASETS[DATASET_NAME]["test"]

        dts = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='train',
                   csv_path_train=csv_train, csv_path_test=csv_test,
                   graph_type=GRAPH_TYPE, temporal_type=TEMPORAL_TYPE,
                   window_size=WINDOW_SIZE, window_step=WINDOW_STEP)

        dts_test = JAAD(root='data', transform=T.Compose([T.ToUndirected()]), mode='test',
                        csv_path_train=csv_train, csv_path_test=csv_test,
                        graph_type=GRAPH_TYPE, temporal_type=TEMPORAL_TYPE,
                        window_size=WINDOW_SIZE, window_step=WINDOW_STEP)

        train_size = int(0.8 * len(dts))
        val_size = len(dts) - train_size

        train_loader = DataLoader(dts[:train_size], batch_size=GLOBAL_BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader   = DataLoader(dts[train_size:], batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=4)
        test_loader  = DataLoader(dts_test, batch_size=GLOBAL_BATCH_SIZE, shuffle=False, num_workers=4)

        model = GraphLevelGNN(
            c_in=C_IN, c_out=C_OUT, c_hidden=C_HIDDEN,
            dp_rate_linear=DP_RATE_LINEAR, dp_rate=DP_RATE,
            num_layers=NUM_LAYERS, layer_name=MODEL_NAME
        )

        wandb_logger = WandbLogger(
            project=WANDB_PROJECT,
            name=f"{MODEL_NAME}_{DATASET_NAME}_seed_{seed}",
            log_model=False
        )

        trainer = pl.Trainer(
            default_root_dir=os.path.join(CHECKPOINT_PATH, f"seed_{seed}"),
            callbacks=[ModelCheckpoint(save_weights_only=True, monitor="val_acc", mode="max"),
                       EarlyStopping(monitor="val_loss", patience=3, mode="min")],
            accelerator="gpu",
            devices=1,
            max_epochs=GLOBAL_MAX_EPOCHS,
            enable_progress_bar=True,
            logger=wandb_logger
        )

        trainer.fit(model, train_loader, val_loader)
        best_model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

        test_result = trainer.test(best_model, dataloaders=test_loader, verbose=False)
        test_acc = test_result[0]["test_acc"]

        print(f"Seed {seed}: {test_acc:.4f}")
        results.append({"seed": seed, "test_acc": test_acc})

        wandb.log({"test_acc": test_acc})
        wandb.finish()

    #Estadísticas
    accs = np.array([r["test_acc"] for r in results])
    print("Media:", accs.mean())
    print("STD:", accs.std())
    print("Rango:", accs.max() - accs.min())
"""

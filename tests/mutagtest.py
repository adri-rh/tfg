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

CHECKPOINT_PATH = "./checkpoints"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}

GLOBAL_batch_size = 100
GLOBAL_max_epochs = 50

class GNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, num_layers=2, layer_name="GCN", dp_rate=0.1, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of the output features. Usually number of classes in classification
            num_layers - Number of "hidden" graph layers
            layer_name - String of the graph layer to use
            dp_rate - Dropout rate to apply throughout the network
            kwargs - Additional arguments for the graph layer (e.g. number of heads for GAT)
        """
        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        for l_idx in range(num_layers-1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
        """
        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x
    
class GraphGNNModel(nn.Module):

    def __init__(self, c_in, c_hidden, c_out, dp_rate_linear=0.5, **kwargs):
        """
        Inputs:
            c_in - Dimension of input features
            c_hidden - Dimension of hidden features
            c_out - Dimension of output features (usually number of classes)
            dp_rate_linear - Dropout rate before the linear layer (usually much higher than inside the GNN)
            kwargs - Additional arguments for the GNNModel object
        """
        super().__init__()
        self.GNN = GNNModel(c_in=c_in,
                            c_hidden=c_hidden,
                            c_out=c_hidden, # Not our prediction output yet!
                            **kwargs)
        self.head = nn.Sequential(
            nn.Dropout(dp_rate_linear),
            nn.Linear(c_hidden, c_out)
        )

    def forward(self, x, edge_index, batch_idx):
        """
        Inputs:
            x - Input features per node
            edge_index - List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            batch_idx - Index of batch element for each node
        """
        x = self.GNN(x, edge_index)
        x = geom_nn.global_mean_pool(x, batch_idx) # Average pooling
        x = self.head(x)
        return x

    
class GraphLevelGNN(pl.LightningModule):

    def __init__(self, **model_kwargs):
        super().__init__()
        # Saving hyperparameters
        self.save_hyperparameters()
        # self.example_input_array = torch.Tensor(6, )

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
        # optimizer = optim.AdamW(self.parameters(), lr=1e-2, weight_decay=0.0) # High lr because of small dataset and small model
        optimizer = optim.AdamW(self.parameters(), lr=1e-3) # High lr because of small dataset and small model
        # optimizer = optim.AdamW(self.parameters(), lr=1e-8, weight_decay=0.0) # High lr because of small dataset and small model
        # optimizer = optim.AdamW(self.parameters(), lr=1e-6, weight_decay=0.0) # High lr because of small dataset and small model

        # return optimizer

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
        self.log('train_acc', acc, prog_bar=True, batch_size = GLOBAL_batch_size)

        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="val")
        self.log('val_acc', acc, prog_bar=True, batch_size=GLOBAL_batch_size)

    def test_step(self, batch, batch_idx):
        _, acc = self.forward(batch, mode="test")
        self.log('test_acc', acc, prog_bar=True, batch_size=GLOBAL_batch_size)

if __name__ == '__main__':
    DATASET_PATH = './data'
    tu_dataset = datasets.TUDataset(root=DATASET_PATH, name="MUTAG")
    torch.manual_seed(42)
    tu_dataset.shuffle()
    MUTAG_train_dataset = tu_dataset[:150]
    MUTAG_test_dataset = tu_dataset[150:]
    MUTAG_graph_train_loader = DataLoader(MUTAG_train_dataset, batch_size=1, shuffle=True)
    MUTAG_graph_val_loader = DataLoader(MUTAG_test_dataset, batch_size=1)  # Additional loader if you want to change to a larger dataset
    MUTAG_graph_test_loader = DataLoader(MUTAG_test_dataset, batch_size=1)
    
    num_node_features = tu_dataset.num_node_features
    num_classes = tu_dataset.num_classes
    
    pl_model = GraphLevelGNN(
        c_in=num_node_features, 
        c_hidden=64, 
        c_out=num_classes,
        layer_name="GCN", 
        num_layers=2
    )
    
    model_name = "GraphConv" 
    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
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
    trainer.fit(pl_model, train_dataloaders=MUTAG_graph_train_loader, val_dataloaders=MUTAG_graph_val_loader)
    
    test_result = trainer.test(pl_model, dataloaders=MUTAG_graph_test_loader)
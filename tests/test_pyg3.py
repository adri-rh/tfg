from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='/tmp/Cora', name='Cora')
print(dataset)

print(len(dataset))

print(dataset.num_classes)

print(dataset.num_node_features)

data = dataset[0]
print(data)

print(data.is_undirected())

print(data.train_mask.sum().item())

print(data.val_mask.sum().item())

print(data.test_mask.sum().item())

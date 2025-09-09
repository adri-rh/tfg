from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
print(dataset)

print(len(dataset))

print(dataset.num_classes)

print(dataset.num_node_features)

data = dataset[0]
print(data)

print(data.is_undirected())

train_dataset = dataset[:540]
print(train_dataset)

test_dataset = dataset[540:]
print(test_dataset)

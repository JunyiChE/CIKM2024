import os
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from transformers import BertTokenizer, AdamW
from model import CrossAttentionFusionModel, GCN
import networkx as nx
from torch_geometric.data import Data


def load_tokenized_data(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data



def convert_to_pyg_data(graph):
    # Map node names to continuous integers if they aren't already
    mapping = {node: i for i, node in enumerate(graph.nodes)}
    relabeled_graph = nx.relabel_nodes(graph, mapping)

    # Now, convert the edges to a tensor
    edge_list = [[source, target] for source, target in relabeled_graph.edges()]
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    data = Data(edge_index=edge_index)  # , x=x if node features are included
    return data


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('data/Twitter16/graph.pkl', 'rb') as f:
        graph = pickle.load(f)
    pyg_data = convert_to_pyg_data(graph).to(device)
    hidden_dims = 768
    out_features = 2
    model = CrossAttentionFusionModel(num_nodes=len(graph),
                                      hidden_dims=hidden_dims,
                                      out_feats=out_features,
                                      bert_pretrained_model_name='bert-base-uncased').to(device)
    uid_to_index = {uid: i for i, uid in enumerate(graph.nodes())}
    train_dataset = load_tokenized_data('data/Twitter16/train_dataset.pkl')
    val_dataset = load_tokenized_data('data/Twitter16/val_dataset.pkl')
    test_dataset = load_tokenized_data('data/Twitter16/test_dataset.pkl')
    # Extract 'tid' from each dataset
    train_tids = set(item['tid'] for item in train_dataset)
    val_tids = set(item['tid'] for item in val_dataset)
    test_tids = set(item['tid'] for item in test_dataset)

    # Check for overlaps
    train_val_overlap = train_tids.intersection(val_tids)
    train_test_overlap = train_tids.intersection(test_tids)
    val_test_overlap = val_tids.intersection(test_tids)

    # Verify no overlap
    if not train_val_overlap:
        print("------------------No data leakage between train and validation sets.------------------")
    if not train_test_overlap:
        print("------------------No data leakage between train and test sets.------------------")
    if not val_test_overlap:
        print("------------------No data leakage between validation and test sets.------------------")
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    num_epochs = 100
    param_names = {param: name for name, param in model.named_parameters()}
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)
    # Print out the parameter names and sizes that the optimizer will update
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                print(param_names[p], p.size())
    best_acc = 0
    best_pre = 0
    best_rec = 0
    best_f1_0 = 0
    best_f1_1 = 0
    for epoch in range(num_epochs):
        # train_loss = train(model, train_loader,optimizer,uid_to_index,pyg_data)
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            # print(texts)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            uids = batch['uid']
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, uids, uid_to_index, pyg_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        model.eval()
        val_loss = 0
        total, correct = 0, 0
        all_predictions = []
        all_labels = []

        for batch in tqdm(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            uids = batch['uid']
            tids = batch['tid']

            outputs = model(input_ids, attention_mask, uids, uid_to_index, pyg_data)
            val_loss_temp = criterion(outputs, labels)
            val_loss += val_loss_temp.item()
            predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

            labels = labels.cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            correct += (predictions == labels).sum()
            total += labels.size
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        accuracy = correct / total
        print(
            f"Epoch {epoch + 1}, Train Loss: {train_loss}, Val Loss: {val_loss / len(val_loader)}, Val Acc :{accuracy}, Pre: {np.mean(precision)}, Rec:{np.mean(recall)}")
        for i, score in enumerate(f1):
            print(f"Label {i} F1 Score: {score:.4f}")
        if accuracy > best_acc:
            best_acc = accuracy
            best_pre = np.mean(precision)
            best_rec = np.mean(recall)
            best_f1_0 = f1[0]
            best_f1_1 = f1[1]
            print(f"Better checkpoint obtained")
            torch.save(model.state_dict(), 'model/best_model_checkpoint.pth')
        test_loss = 0
        total, correct = 0, 0
        all_predictions = []
        all_labels = []

        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            uids = batch['uid']
            tids = batch['tid']

            outputs = model(input_ids, attention_mask, uids, uid_to_index, pyg_data)
            test_loss_temp = criterion(outputs, labels)
            test_loss += test_loss_temp.item()
            predictions = torch.argmax(outputs, dim=-1).cpu().numpy()

            labels = labels.cpu().numpy()
            all_predictions.extend(predictions)
            all_labels.extend(labels)
            correct += (predictions == labels).sum()
            total += labels.size
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average=None)
        accuracy = correct / total
        print(
            f"Epoch {epoch + 1},Test Loss: {test_loss / len(test_loader)}, Test Acc :{accuracy}, Pre: {np.mean(precision)}, Rec:{np.mean(recall)}")
        for i, score in enumerate(f1):
            print(f"Label {i} F1 Score: {score:.4f}")
    print(
        f"Training down, best acc: {best_acc}, best re: {best_pre}, best rec: {best_rec}, best f1-non-rumor: {best_f1_0} best f1-rumor: {best_f1_1}")


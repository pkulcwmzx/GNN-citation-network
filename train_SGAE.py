import torch
import argparse
import pdb
import time
import copy
import os
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from models.GraphSAGE import GraphSAGE
from sklearn import metrics
from torch.optim import Adam
from torch.nn import functional as F
def train():
    # get the parameters
    args = get_args()

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.cuda else 'cpu')
    if args.dataset == "Cora":
        dataset = Planetoid(root='./dataset', name='Cora', transform=T.NormalizeFeatures())
    elif args.dataset == "CiteSeer":
        dataset = Planetoid(root='./dataset', name='CiteSeer', transform=T.NormalizeFeatures())
    elif args.dataset == "PubMed":
        dataset = Planetoid(root='./dataset', name='PubMed', transform=T.NormalizeFeatures())
    else:
        print("Dataset Error!!")

    data = dataset[0].to(device)

    # create the model and optimizer
    model = GraphSAGE(dataset.num_features, args.hidden_dim, dataset.num_classes, args.normalize).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # the information which need to be recorded
    start_time = time.time()
    bad_counter = 0
    best_valid_f1 = 0.0
    best_epoch = 0
    least_loss = float("inf")
    best_model = None

    # beging training
    for epoch in range(args.epochs):
        # the steps of training
        model.train()
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
        current_loss = loss.item()
        loss.backward()
        optimizer.step()

        # validate current model
        f1 = validate(model, data)
        print('Epoch: {:04d}'.format(epoch + 1), 'f1: {:.4f}'.format(f1), 'loss: {:.4f}'.format(current_loss))

        # save the model if it access the minimum loss in current epoch
        if current_loss < least_loss:
            least_loss = current_loss
            best_epoch = epoch + 1
            best_valid_f1 = f1
            best_model = copy.deepcopy(model)
            bad_counter = 0
        else:
            bad_counter += 1

        # early stop
        if bad_counter >= args.patience:
            break

    print("Optimization Finished!")
    used_time = time.time() - start_time
    print("Total epochs: {:2d}".format(best_epoch + 100))
    print("Best epochs: {:2d}".format(best_epoch))
    # print("Time for each epoch: {:.2f}s".format(used_time / (best_epoch + args.patience)))
    # print("Best epoch's validate f1: {:.2f}".format(best_valid_f1 * 100))
    # test the best trained model
    test(best_model, data)
    print("Total time elapsed: {:.2f}s".format(used_time))


def validate(model, data):
    model.eval()
    output = model(data)
    pred = output[data.val_mask].max(1)[1].cpu().numpy()
    gold = data.y[data.val_mask].cpu().numpy()

    return metrics.f1_score(gold, pred, average='macro')


def test(model, data):
    model.eval()
    output = model(data)
    pred = output[data.test_mask].max(1)[1].cpu().numpy()
    gold = data.y[data.test_mask].cpu().numpy()

    print("Test set results:",
          "accu= {:.2f}".format(metrics.accuracy_score(gold, pred) * 100),
          "macro_p= {:.2f}".format(metrics.precision_score(gold, pred, average='macro') * 100),
          "macro_r= {:.2f}".format(metrics.recall_score(gold, pred, average='macro') * 100),
          "macro_f1= {:.2f}".format(metrics.f1_score(gold, pred, average='macro') * 100),
          "micro_f1= {:.2f}".format(metrics.f1_score(gold, pred, average='micro') * 100))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora', help='which dataset to be used')
    parser.add_argument('--cuda', action='store_true', default=True, help='Disables CUDA training.')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
    parser.add_argument('--hidden_dim', type=int, default=16, help='Number of hidden units.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--normalize', default=True, help='Number of hidden units.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')

    return parser.parse_args()

if __name__ == '__main__':
    train()
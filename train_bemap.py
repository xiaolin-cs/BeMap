import os
from os.path import *
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

import dgl
from utils import *
# from pygcn.models import *
from models import *
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=53, help='Random seed.')
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--beta', type=float, default=0.25)
parser.add_argument('--lam', type=float, default=0.5)
parser.add_argument('--save_num', type=int, default=4)
parser.add_argument('--dataset', type=str, default='pokec_z',
                    choices=['pokec_z', 'nba', 'bail', 'credit'])
parser.add_argument('--model', type=str, default='gcn', choices=['gcn', 'gat'])

args = parser.parse_known_args()[0]
args.cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

fair_require = {
    'pokec_z': [0.025, 0.025],
    'nba': [0.07, 0.07],
    'bail': [0.05, 0.05],
    'credit': [0.11, 0.11],
}


def main():
    print(args)
    # Load pokec data ../../FairGNN/dataset/pokec
    if args.dataset == 'pokec_z':
        dataset = "region_job"
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"
        label_number = 500
        sens_number = 200
        seed = 20
        path = "./dataset/pokec/"
        test_idx = False

        adj, features, labels, idx_train, \
        idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset, sens_attr, predict_attr, path=path,
                                                             label_number=label_number, sens_number=sens_number,
                                                             seed=seed, test_idx=test_idx)

    # Load nba data
    elif args.dataset == 'nba':
        dataset = 'nba'
        sens_attr = "country"
        predict_attr = "SALARY"
        label_number = 100
        sens_number = 50
        seed = 20
        path = "./dataset/NBA"
        test_idx = True

        adj, features, labels, idx_train, \
        idx_val, idx_test, sens, idx_sens_train = load_pokec(dataset, sens_attr, predict_attr, path=path,
                                                             label_number=label_number, sens_number=sens_number,
                                                             seed=seed, test_idx=test_idx)

    # Load credit data
    elif args.dataset == 'credit':
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_credit('credit',
                                                                                                path='./dataset/credit',
                                                                                                label_number=6000)


    # Load bail data
    elif args.dataset == 'bail':
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_bail('bail',
                                                                                              path='./dataset/bail',
                                                                                              label_number=100)


    else:
        print('Invalid dataset name!!')
        exit(0)

    features = feature_norm(features)

    if args.model == 'gcn':
        model = BeMap_GCN(
            nfeat=features.shape[1], nhid=args.hidden, nclass=1, dropout=args.dropout, graph=adj, attrs=sens, lam=args.lam,
            beta=args.beta, save_num=args.save_num, use_cuda=args.cuda, norm='both')
    elif args.model == 'gat':
        model = BeMap_GAT(
            nfeat=features.shape[1], nhid=args.hidden, nclass=1, dropout=args.dropout, graph=adj, attrs=sens, lam=args.lam,
            beta=args.beta, save_num=args.save_num, use_cuda=args.cuda)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()
        features = features.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()

    # Train model
    t_total = time.time()
    best_acc = 0
    best_roc = 0
    best_sp = +np.inf
    best_eo = +np.inf
    fair_req = fair_require[args.dataset]
    for epoch in range(args.epochs):
        train(epoch, model, optimizer, features, labels, idx_train)
        acc, roc, sp, eo = test(model, labels, sens, features, idx_test, epoch)
        if sp < fair_req[0] and eo < fair_req[1]:
            if acc > best_acc:
                best_acc = acc
                best_roc = roc
                best_sp = sp
                best_eo = eo
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    print('--- Test Results ---')
    print('acc: {:.4f}'.format(best_acc))
    print('roc: {:.4f}'.format(best_roc))
    print('sp: {:.4f}'.format(best_sp))
    print('eo: {:.4f}'.format(best_eo))


def train(epoch, model, optimizer, features, labels, idx_train):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, epoch)
    loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
    acc_train = accuracy_logit(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(model, labels, sens, features, idx_test, epoch):
    model.eval()
    output = model(features, -1)
    acc_test = accuracy_logit(output[idx_test], labels[idx_test])
    roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())

    val_y = labels[idx_test].cpu().numpy()
    idx_s0 = sens.numpy()[idx_test.cpu().numpy()] == 0
    idx_s1 = sens.numpy()[idx_test.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx_test].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

    info = "Test: {:04d} accuracy: {:.4f} roc_test: {:.4f} parity: {:.4f} " \
           "equality: {:.4f}".format(
        epoch + 1, acc_test.item(), roc_test, parity, equality
    )

    print(info)
    return acc_test, roc_test, parity, equality


if __name__ == '__main__':
    main()

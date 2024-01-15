import os
import itertools
from library_data import *
from library_models import *
import numpy as np
from evaluate_model import evaluate_results, EarlyStopper, compute_accuracy
import argparse
import datetime


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=0, type=int, help='ID of the gpu to run on.')
parser.add_argument('--epochs', default=200, type=int, help='Number of epochs to train the model')
parser.add_argument('--num_weeks', default=57, type=int, help='Number of weeks of data with labels.')
parser.add_argument('--gnn_hidden_dim', default=500, type=int,
                    help='Number of dimensions of GNN node embeddings.')
parser.add_argument('--rnn_hidden_dim', default=500, type=int,
                    help='Number of dimensions of LSTM hidden state.')
parser.add_argument('--fcn_dim', default=256, type=int, help='Number of dimensions of the MLP hidden layer.')
parser.add_argument('--val_num_graphs', default=3, type=int, help='Number of validation graphs.')
parser.add_argument('--test_num_graphs', default=5, type=int, help='Number of test graphs.')
parser.add_argument('--num_estimators', default=2, type=int, help='Number of members in ensemble learning.')
parser.add_argument('--lookback', default=2, type=int, help='Lookback value in temporal encoder.')
parser.add_argument('--learning_rate', default=0.0005, type=float, help='Learning rate of Adam optimizer.')
parser.add_argument('--dropout', default=0.2, type=float, help='Dropout probability in MLP.')
parser.add_argument('--gnn_dropout', default=0.1, type=float, help='Feat dropout probability in GNN layer.')
parser.add_argument('--use_gru', default=False, type=bool,
                    help='True if using GRU as the recurrent unit. False otherwise. By default, set to False.')
args = parser.parse_args()

os.environ['DGLBACKEND'] = 'pytorch'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

epochs = args.epochs
learning_rate = args.learning_rate
gnn_hidden_dim = args.gnn_hidden_dim
rnn_hidden_dim = args.rnn_hidden_dim
fcn_dim = args.fcn_dim
lookback = args.lookback
num_weeks = args.num_weeks
dropout = args.dropout
gnn_feat_drop = args.gnn_dropout
use_gru = args.use_gru
use_early_stop = True

val_num_weeks = args.val_num_graphs
test_num_weeks = args.test_num_graphs


BASE_SAVE_DIR = f'./models/{args.num_weeks}/val{val_num_weeks}_test{test_num_weeks}'
if not os.path.exists(BASE_SAVE_DIR):
    os.makedirs(BASE_SAVE_DIR)

dataset = TwitterDataset(num_weeks, stride=1, use_week_count=num_weeks)

def compute_loss( unlink_pos_score, unlink_neg_score):

    unlink_scores = torch.cat([unlink_pos_score, unlink_neg_score])
    unlink_labels = torch.cat([torch.ones(unlink_pos_score.shape[0]), torch.zeros(unlink_neg_score.shape[0])]).to(
        'cuda')
    loss = F.binary_cross_entropy_with_logits(unlink_scores, unlink_labels)

    return loss


def get_window_data(current_batch):

    graphs = [graph.to('cuda') for graph in current_batch[0]]

    unfollow_splits = current_batch[1][-1]
    feat_list = last_graph.ndata['feat'].to('cuda')
    return graphs, feat_list, unfollow_splits


def train():
    unlink_pos_score = unlink_neg_score = None
    val_unlink_pos_score = val_unlink_neg_score = None

    model = DynGraphSAGE(last_graph.ndata['feat'].shape[1], gnn_hidden_dim, lookback=lookback,
                        feat_drop=gnn_feat_drop).to(
        'cuda')
    if lookback != 1:
        sequence_model = LSTMModel(gnn_hidden_dim, rnn_hidden_dim, fcn_dim=fcn_dim, num_layers=1,
                                   lookback=lookback,
                                   dropout=dropout, use_gru=use_gru).to('cuda')
    else:
        sequence_model = FCN_MODEL(gnn_hidden_dim, fcn_dim, lookback=lookback, dropout=dropout).to('cuda')

    early_stopper = EarlyStopper(patience=2, min_delta=0.001)


    optimizer = torch.optim.Adam(itertools.chain(model.parameters(), sequence_model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    for e in range(epochs):

        total_loss, loss = 0, 0

        for item in train_batch:
            model.train()
            sequence_model.train()
            graphs, feat_list, unfollow_splits = get_window_data(item)

            # forward
            h_list = model(graphs, feat_list)

            unlink_pos_score = sequence_model(unfollow_splits[0].to('cuda'), h_list)
            unlink_neg_score = sequence_model(unfollow_splits[1].to('cuda'), h_list)
            loss = compute_loss(unlink_pos_score, unlink_neg_score)

            total_loss += loss.item()
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        sequence_model.eval()
        val_total_loss, val_loss = 0, 0
        val_avg_acc = []
        for item in val_batch:
            with torch.no_grad():
                val_graphs, val_feat_list, val_unfollow_splits = get_window_data(item)
                h_list = model(val_graphs, val_feat_list)

                val_unlink_pos_score = sequence_model(val_unfollow_splits[0].to('cuda'), h_list)
                val_unlink_neg_score = sequence_model(val_unfollow_splits[1].to('cuda'), h_list)
                val_acc = compute_accuracy(val_unlink_pos_score, val_unlink_neg_score)
                val_avg_acc.append(val_acc)
                val_loss = compute_loss(val_unlink_pos_score, val_unlink_neg_score)
                val_total_loss += val_loss.item()

        if use_early_stop & early_stopper.early_stop(val_total_loss / len(val_batch)):
            print(
                "In epoch {}, loss: {:.5f}, val loss: {:.5f}, val acc: {:.4f}".format(e, total_loss / len(train_batch),
                                                                                      val_total_loss / len(val_batch),
                                                                                      sum(val_avg_acc) / len(
                                                                                          val_avg_acc)))
            early_stop = True
            break
        elif (not use_early_stop) or (val_total_loss / len(val_batch) <= early_stopper.min_validation_loss):
            torch.save(model.cpu().state_dict(), gnn_model_path)
            torch.save(sequence_model.cpu().state_dict(), seq_model_path)
            model.cuda()
            sequence_model.cuda()

        if (e % (epochs / 20) == 0):
            print("In epoch {}, loss: {:.5f}, val loss: {:.5f}, val acc: {:.4f}".format(e, total_loss / len(train_batch),
                                                                                    val_total_loss / len(val_batch),
                                                                                    sum(val_avg_acc) / len(
                                                                                        val_avg_acc)))

def load_model():
    model = DynGraphSAGE(last_graph.ndata['feat'].shape[1], gnn_hidden_dim, lookback=lookback,
                        feat_drop=gnn_feat_drop).to(
        'cuda')
    if lookback != 1:
        sequence_model = LSTMModel(gnn_hidden_dim, rnn_hidden_dim, fcn_dim=fcn_dim, num_layers=1,
                                   lookback=lookback,
                                   dropout=dropout, use_gru=use_gru).to('cuda')
    else:
        sequence_model = FCN_MODEL(gnn_hidden_dim, fcn_dim, lookback=lookback, dropout=dropout).to('cuda')
    model.load_state_dict(torch.load(gnn_model_path))
    sequence_model.load_state_dict(torch.load(seq_model_path))
    model.eval()
    sequence_model.eval()
    return model, sequence_model


def evaluate_model(batch, type):
    accuracy = []
    val_ps = []
    val_ns = []
    test_ps = []
    test_ns = []
    for item in batch:
        graphs, feat_list, unfollow_splits = get_window_data(item)

        h_list = model(graphs, feat_list)

        unlink_pos_score = sequence_model(unfollow_splits[0].to('cuda'), h_list)
        unlink_neg_score = sequence_model(unfollow_splits[1].to('cuda'), h_list)

        if type == 'Validation':
            val_ps.append(unlink_pos_score)
            val_ns.append(unlink_neg_score)
        else:
            test_ps.append(unlink_pos_score)
            test_ns.append(unlink_neg_score)

        if type == 'Validation':
            val_pos_scores.append(val_ps)
            val_neg_scores.append(val_ns)
        else:
            test_pos_scores.append(test_ps)
            test_neg_scores.append(test_ns)

np.random.seed(42)
torch.manual_seed(42)

test_pos_scores = []
test_neg_scores = []

val_pos_scores = []
val_neg_scores = []
sigmoid_fn = nn.Sigmoid()
currdate = str(datetime.datetime.today().strftime('%Y%m%d%H%M%S'))


print(f'lookback #{lookback}')
SAVE_DIR = BASE_SAVE_DIR+ f'/w{lookback}'
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
for s in range(args.num_estimators):
    print(f'estimator #{s}')
    stride = 1
    all_data = DynamicDataset(dataset, lookback, stride=stride)

    last_graph = all_data[len(all_data)][0][lookback - 1]

    train_num_weeks = len(all_data) - val_num_weeks - test_num_weeks + 1

    val_batch = all_data[train_num_weeks:train_num_weeks + val_num_weeks]
    test_batch = all_data[train_num_weeks + val_num_weeks: len(all_data) + 1]
    train_batch = all_data[0:train_num_weeks]


    link_pos_score = link_neg_score = unlink_pos_score = unlink_neg_score = None
    val_link_pos_score = val_link_neg_score = val_unlink_pos_score = val_unlink_neg_score = None

    gnn_model_path = SAVE_DIR + f'/iter{s}_gp{args.gpu}_gnn_{currdate}.pth'
    seq_model_path = SAVE_DIR + f'/iter{s}_gp{args.gpu}_pred_{currdate}.pth'

    train()

    model, sequence_model = load_model()


    with torch.no_grad():
        evaluate_model(val_batch, 'Validation')
        evaluate_model(test_batch, 'Test')

accuracy = []
auc_list = []
prec = []
rec = []
f1 = []
supp = 0
reports = []
for i in range(val_num_weeks):

    unlink_pos_score = torch.stack([item[i] for item in val_pos_scores], dim=1)
    unlink_pos_score = torch.mean(unlink_pos_score, dim=1)
    unlink_neg_score = torch.mean(torch.stack([item[i] for item in val_neg_scores], dim=1), dim=1)
    acc, precision, recall, f1_score, auc, report = evaluate_results(unlink_pos_score, unlink_neg_score, f'Validation Unfollow', activation=sigmoid_fn)
    support = len(unlink_pos_score) * 2
    accuracy.append(acc * support)
    auc_list.append(auc * support)
    prec.append(precision * support)
    rec.append(recall * support)
    f1.append(f1_score * support)
    supp += support
    reports.append(str(report))


print(f'Validation Weighted Average Accuracy: {sum(accuracy) / supp:.4f}')
print(f'Validation Weighted Average Precision: {sum(prec) / supp:.4f}')
print(f'Validation Weighted Average Recall: {sum(rec) / supp:.4f}')
print(f'Validation Weighted Average F1_score: {sum(f1) / supp:.4f}')
print(f'Validation Weighted Average AUC: {sum(auc_list) / supp:.4f}')
accuracy = []
auc_list = []
prec = []
rec = []
f1 = []
supp = 0
reports = []
for i in range(test_num_weeks):
    unlink_pos_score = torch.stack([item[i] for item in test_pos_scores], dim=1)
    unlink_pos_score = torch.mean(unlink_pos_score, dim=1)
    unlink_neg_score = torch.mean(torch.stack([item[i] for item in test_neg_scores], dim=1), dim=1)
    acc, precision, recall, f1_score, auc, report = evaluate_results(unlink_pos_score, unlink_neg_score, f'Test Unfollow', activation=sigmoid_fn)
    support = len(unlink_pos_score) * 2
    accuracy.append(acc * support)
    auc_list.append(auc * support)
    prec.append(precision * support)
    rec.append(recall * support)
    f1.append(f1_score * support)
    supp += support
    reports.append(str(report))


print(f'Test Weighted Average Accuracy: {sum(accuracy) / supp:.4f}')
print(f'Test Weighted Average Precision: {sum(prec) / supp:.4f}')
print(f'Test Weighted Average Recall: {sum(rec) / supp:.4f}')
print(f'Test Weighted Average F1_score: {sum(f1) / supp:.4f}')
print(f'Test Weighted Average AUC: {sum(auc_list) / supp:.4f}')

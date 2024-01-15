import torch
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def compute_auc(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return roc_auc_score(labels, scores)


def compute_accuracy(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    pred_labels = (scores >= 0.5).astype(int)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return accuracy_score(labels, pred_labels)

def compute_precision(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    pred_labels = (scores >= 0.5).astype(int)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return precision_score(labels, pred_labels)

def compute_recall(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    pred_labels = (scores >= 0.5).astype(int)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return recall_score(labels, pred_labels)

def compute_f1_score(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    pred_labels = (scores >= 0.5).astype(int)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return f1_score(labels, pred_labels)


def compute_classification_report(pos_score, neg_score):
    scores = torch.cat([pos_score, neg_score]).detach().cpu().numpy()
    pred_labels = (scores >= 0.5).astype(int)
    labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
    return classification_report(labels, pred_labels, labels=[0, 1])

def evaluate_results(pos_score, neg_score, result_type, activation):
    print(f'\n{result_type} Results:')
    auc = compute_auc(pos_score, neg_score)
    print("AUC: {:.2f}".format(auc))

    pos_score = activation(torch.unsqueeze(pos_score, 1))
    neg_score = activation(torch.unsqueeze(neg_score, 1))

    precision = compute_precision(pos_score, neg_score)
    recall = compute_recall(pos_score, neg_score)
    f1_score = compute_f1_score(pos_score, neg_score)
    acc = compute_accuracy(pos_score, neg_score)
    print("Classification report:\n", compute_classification_report(pos_score, neg_score))

    return acc, precision, recall, f1_score, auc, compute_classification_report(pos_score, neg_score)
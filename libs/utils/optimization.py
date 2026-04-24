from sklearn.metrics import confusion_matrix, matthews_corrcoef
from sklearn.metrics import roc_auc_score, auc, accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import torch
from tqdm import tqdm

import sys 
sys.path.append("../") 
from libs.utils.utils import move_dict_to_device

def compute_evaluation_scores(label_list, prediction_list, score_list):
    accuracy = accuracy_score(y_true=label_list, y_pred=prediction_list)
    tn, fp, fn, tp = confusion_matrix(label_list, prediction_list, labels=[0, 1]).ravel().tolist()
    if tp + fn == 0:
        sensitivity = 0
    else:
        sensitivity = tp / (tp + fn)
    if tn + fp == 0:
        specificity = 0
    else:
        specificity = tn / (tn + fp)
    precision = precision_score(y_true=label_list, y_pred=prediction_list)
    recall = recall_score(y_true=label_list, y_pred=prediction_list)
    f1 = f1_score(y_true=label_list, y_pred=prediction_list)
    roc_auc = roc_auc_score(label_list, score_list)
    prec, reca, _ = precision_recall_curve(label_list, score_list)
    aupr = auc(reca, prec)
    mcc = matthews_corrcoef(label_list, prediction_list)
    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "aupr": aupr,
        "mcc": mcc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def validate(model, val_loader, loss_function=None, device="cpu"):
    label_list = []
    prediction_list = []
    score_list = []
    if loss_function is not None:
        loss_sum = 0

    with torch.no_grad():
        model.eval()
        
        for batchDict in tqdm(val_loader, desc="Validating"):

            batchDict = move_dict_to_device(batchDict, device)
            labels = batchDict["label"]
            prediction, att_scores, raw_weights = model(batchDict)
            prediction_list += torch.argmax(prediction, dim=1).detach().cpu().tolist()
            score_list += torch.softmax(prediction, dim=1).detach().cpu()[:, 1].tolist()  
            label_list += labels.detach().cpu().tolist()
            if loss_function is not None:
                loss = loss_function(prediction, labels)
                loss_sum += loss.item()

    performance = compute_evaluation_scores(label_list, prediction_list, score_list)
    if loss_function is not None:
        performance["loss"] = loss_sum / len(label_list)

    return performance


def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    label_list = []
    prediction_list = []
    score_list = []
    loss_sum = 0
    for batchDict in tqdm(train_loader, desc="Training"):

        batchDict = move_dict_to_device(batchDict, device)

        labels = batchDict["label"]
        prediction, att_scores, raw_weights = model(batchDict)
        loss = loss_function(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        prediction_list += torch.argmax(prediction, dim=1).detach().cpu().tolist()
        score_list += torch.softmax(prediction, dim=1).detach().cpu()[:, 1].tolist()  
        label_list += labels.detach().cpu().tolist()
        loss_sum += loss.item()

    performance = compute_evaluation_scores(label_list, prediction_list, score_list)
    performance["loss"] = loss_sum / len(label_list)
    
    return performance
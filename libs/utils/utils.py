import yaml
import os
import torch



# load config yaml
def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config




class Logger(object):
    def __init__(self, log_path):
        self.log_path = log_path
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
    def write(self, message):
        print(message)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message)


def log_format(train_performance, val_performance=None, epoch=0, independent_performance=None, external_performance=None):
    """
    {
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
    """
    msg = "\n *********************************************************\n"
    msg += "Epoch: {}: \n".format(epoch)
    msg += "   Train: Loss*1000: {:.4f} | roc_auc: {: .4f} | aupr: {: .4f} | accuracy: {: .4f} | \n          sensitivity: {: .4f} | specificity: {: .4f} | precision: {: .4f} | recall: {: .4f} | \n          f1: {: .4f} | mcc: {: .4f} |  \n          tn: {: .4f} | fp: {: .4f} | fn: {: .4f} | tp: {: .4f} \n".format(
        train_performance["loss"] * 1000,
        train_performance["roc_auc"],
        train_performance["aupr"],
        train_performance["accuracy"],
        train_performance["sensitivity"],
        train_performance["specificity"],
        train_performance["precision"],
        train_performance["recall"],
        train_performance["f1"],
        train_performance["mcc"],
        train_performance["tn"],
        train_performance["fp"],
        train_performance["fn"],
        train_performance["tp"],
        )
    if val_performance:
        msg += "   Val: Loss*1000: {:.4f} | roc_auc: {: .4f} | aupr: {: .4f} | accuracy: {: .4f} | \n          sensitivity: {: .4f} | specificity: {: .4f} | precision: {: .4f} | recall: {: .4f} | \n          f1: {: .4f} | mcc: {: .4f} |   \n          tn: {: .4f} | fp: {: .4f} | fn: {: .4f} | tp: {: .4f} \n".format(
            val_performance["loss"] * 1000,
            val_performance["roc_auc"],
            val_performance["aupr"],
            val_performance["accuracy"],
            val_performance["sensitivity"],
            val_performance["specificity"],
            val_performance["precision"],
            val_performance["recall"],
            val_performance["f1"],
            val_performance["mcc"],
            val_performance["tn"],
            val_performance["fp"],
            val_performance["fn"],
            val_performance["tp"],
            )
    
    if independent_performance:
            msg += "   Independent: Loss*1000: {:.4f} | roc_auc: {: .4f} | aupr: {: .4f} | accuracy: {: .4f} | \n          sensitivity: {: .4f} | specificity: {: .4f} | precision: {: .4f} | recall: {: .4f} | \n          f1: {: .4f} | mcc: {: .4f} |   \n          tn: {: .4f} | fp: {: .4f} | fn: {: .4f} | tp: {: .4f} \n".format(
                independent_performance["loss"] * 1000,
                independent_performance["roc_auc"],
                independent_performance["aupr"],
                independent_performance["accuracy"],
                independent_performance["sensitivity"],
                independent_performance["specificity"],
                independent_performance["precision"],
                independent_performance["recall"],
                independent_performance["f1"],
                independent_performance["mcc"],
                independent_performance["tn"],
                independent_performance["fp"],
                independent_performance["fn"],
                independent_performance["tp"],
                )
    if external_performance:
        msg += "   External: Loss*1000: {:.4f} | roc_auc: {: .4f} | aupr: {: .4f} | accuracy: {: .4f} | \n          sensitivity: {: .4f} | specificity: {: .4f} | precision: {: .4f} | recall: {: .4f} | \n          f1: {: .4f} | mcc: {: .4f}  |   \n          tn: {: .4f} | fp: {: .4f} | fn: {: .4f} | tp: {: .4f} \n".format(
                external_performance["loss"] * 1000,
                external_performance["roc_auc"],
                external_performance["aupr"],
                external_performance["accuracy"],
                external_performance["sensitivity"],
                external_performance["specificity"],
                external_performance["precision"],
                external_performance["recall"],
                external_performance["f1"],
                external_performance["mcc"],
                external_performance["tn"],
                external_performance["fp"],
                external_performance["fn"],
                external_performance["tp"],
                )
    msg += " *********************************************************\n"
    return msg



def move_dict_to_device(nested_dict, device):
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            nested_dict[key] = move_dict_to_device(value, device)
        elif isinstance(value, torch.Tensor):
            nested_dict[key] = value.to(device)
        elif isinstance(value, list):
            nested_dict[key] = [
                move_dict_to_device(item, device) if isinstance(item, dict) 
                else item.to(device) if isinstance(item, torch.Tensor) 
                else item 
                for item in value
            ]
    return nested_dict
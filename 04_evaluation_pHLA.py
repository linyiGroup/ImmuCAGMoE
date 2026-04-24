from libs.utils.utils import load_config, Logger, log_format
from libs.dataset import PHLADataset
from torch.utils.data import DataLoader
from libs.models import PHLAModel
from libs.utils.optimization import train_one_epoch, validate

from torch import nn
import torch.optim as optim
import torch
import pandas as pd

    


def valid_pHLA(config, config_model):

    model = PHLAModel(config_model).to(config["device"])
    model.load_state_dict(torch.load(config["modelStatePath"] + "pHLA_model_final.pth"))

    independent_set = PHLADataset(csv_path=config["dataPath"]+"independent_set.csv", config=config)
    independent_loader = DataLoader(independent_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    independent_performance = validate(model, independent_loader, None, config["device"])
    independent_performance = pd.DataFrame([independent_performance], index=["independent"])

    external_set = PHLADataset(csv_path=config["dataPath"]+"external_set.csv", config=config)
    external_loader = DataLoader(external_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    external_performance = validate(model, external_loader, None, config["device"])
    external_performance = pd.DataFrame([external_performance], index=["external"])

    HPV_set = PHLADataset(csv_path=config["dataPath"]+"HPV_set.csv", config=config)
    HPV_loader = DataLoader(HPV_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    HPV_performance = validate(model, HPV_loader, None, config["device"])
    HPV_performance = pd.DataFrame([HPV_performance], index=["HPV"])

    neoantigen_set = PHLADataset(csv_path=config["dataPath"]+"neoantigen_set.csv", config=config)
    neoantigen_loader = DataLoader(neoantigen_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    neoantigen_performance = validate(model, neoantigen_loader, None, config["device"])
    neoantigen_performance = pd.DataFrame([neoantigen_performance], index=["neoantigen"])
    
    return independent_performance, external_performance, HPV_performance, neoantigen_performance


if __name__ == "__main__":

    config = load_config("/linyi/workspace/TCRpHLA/ImmuCAGMoE/configs/config_evaluation_pHLA.yaml")
    config_model = load_config("/linyi/workspace/TCRpHLA/ImmuCAGMoE/configs/config_model.yaml")

    independent_performance, external_performance, HPV_performance, neoantigen_performance \
        = valid_pHLA(config, config_model)
    
    all_performance = pd.concat([independent_performance, external_performance, HPV_performance, neoantigen_performance])
    all_performance = pd.DataFrame(all_performance).round(4)
    all_performance.to_csv(config["logPath"] + "final_performance_all.csv")




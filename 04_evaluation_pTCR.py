from libs.utils.utils import load_config, Logger, log_format
from libs.dataset import PTCRDataset
from torch.utils.data import DataLoader
from libs.models import PTCRModel
from libs.utils.optimization import train_one_epoch, validate

from torch import nn
import torch.optim as optim
import torch
import pandas as pd

    


def valid_pTCR(config, config_model):

    model = PTCRModel(config_model).to(config["device"])
    model.load_state_dict(torch.load(config["modelStatePath"] + "pTCR_model_final.pth"))
    
    independent_set = PTCRDataset(csv_path=config["dataPath"]+"independent_set.csv", config=config)
    independent_loader = DataLoader(independent_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    independent_performance = validate(model, independent_loader, None, config["device"])
    independent_performance = pd.DataFrame([independent_performance], index=["independent"])

    external_set = PTCRDataset(csv_path=config["dataPath"]+"external_set.csv", config=config)
    external_loader = DataLoader(external_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    external_performance = validate(model, external_loader, None, config["device"])
    external_performance = pd.DataFrame([external_performance], index=["external"])

    COVID_set = PTCRDataset(csv_path=config["dataPath"]+"COVID_set.csv", config=config)
    COVID_loader = DataLoader(COVID_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    COVID_performance = validate(model, COVID_loader, None, config["device"])
    COVID_performance = pd.DataFrame([COVID_performance], index=["COVID"])


    return independent_performance, external_performance, COVID_performance


if __name__ == "__main__":

    config = load_config("/linyi/workspace/TCRpHLA/ImmuCAGMoE/configs/config_evaluation_pTCR.yaml")
    config_model = load_config("/linyi/workspace/TCRpHLA/ImmuCAGMoE/configs/config_model.yaml")


    independent_performance, external_performance, COVID_performance \
        = valid_pTCR(config, config_model)
        
    
    all_performance = pd.concat([independent_performance, external_performance, COVID_performance])
    all_performance = pd.DataFrame(all_performance).round(4)
    all_performance.to_csv(config["logPath"] + "final_performance_all.csv")




# nohup python 03_train_pHLA.py > ./results/train/pHLA/nohup.log 2>&1 &

from libs.utils.utils import load_config, Logger, log_format
from libs.dataset import PHLADataset
from torch.utils.data import DataLoader
from libs.models import PHLAModel
from libs.utils.optimization import train_one_epoch, validate

from torch import nn
import torch.optim as optim
import torch
import pandas as pd

    


def train_valid_pHLA(config, config_model):

    logger = Logger(config["logPath"] + "train.log")

    train_file_name = "dataset.csv"
    val_file_name = "independent_set.csv"

    train_set = PHLADataset(csv_path=config["dataPath"]+train_file_name, config=config)
    val_set = PHLADataset(csv_path=config["dataPath"]+val_file_name, config=config)

    train_loader = DataLoader(train_set, batch_size=config["batchSize"], shuffle=True, num_workers=config["numWorker"])
    val_loader = DataLoader(val_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])

    model = PHLAModel(config_model).to(config["device"])
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["learningRate"]))

    logger.write("Start training...\n\n")
    for epoch in range(config["epochs"]):
        train_performance = train_one_epoch(model, train_loader, loss_function, optimizer, config["device"])
        val_performance = validate(model, val_loader, loss_function, config["device"])

        msg = log_format(train_performance, 
                         independent_performance=val_performance, 
                         epoch=epoch)
        logger.write(msg)

        torch.save(model.state_dict(), config["modelStatePath"] + "pHLA_model.pth")

    model.load_state_dict(torch.load(config["modelStatePath"] + "pHLA_model.pth"))
    
    independent_set = PHLADataset(csv_path=config["dataPath"]+"independent_set.csv", config=config)
    independent_loader = DataLoader(independent_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    independent_performance = validate(model, independent_loader, loss_function, config["device"])
    independent_performance = pd.DataFrame([independent_performance], index=["independent"])

    external_set = PHLADataset(csv_path=config["dataPath"]+"external_set.csv", config=config)
    external_loader = DataLoader(external_set, batch_size=config["batchSize"], shuffle=False, num_workers=config["numWorker"])
    external_performance = validate(model, external_loader, loss_function, config["device"])
    external_performance = pd.DataFrame([external_performance], index=["external"])

    return independent_performance, external_performance


if __name__ == "__main__":

    config = load_config("./configs/config_train_pHLA.yaml")
    config_model = load_config("./configs/config_model.yaml")

    independent_performance, external_performance = train_valid_pHLA(config, config_model)
        
    all_performance = pd.concat([independent_performance, external_performance])
    all_performance = pd.DataFrame(all_performance).round(4)
    all_performance.to_csv(config["logPath"] + "performance_all.csv")



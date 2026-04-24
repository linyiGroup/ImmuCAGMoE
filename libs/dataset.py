from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np


import sys 
sys.path.append("../") 
from libs.utils.embedding import Embedder


class PHLADataset(Dataset):
    def __init__(self, csv_path, config):
        self.data = pd.read_csv(csv_path)
        self.peptideList = self.data.peptide.tolist()
        self.HLAList = self.data.HLA.tolist()
        self.labelList = self.data.label.tolist()

        self.embedder = Embedder(
            seqMaxLength=config["seqMaxLength"], 
            esmcEmbeddingDictPath=config["esmcEmbeddingDictPath"],
            atchleyFactorPath=config["atchleyFactorPath"],
            esmcEmbeddingPath=config["esmcEmbeddingPath"])
        
        self.peptideBlosumNum = config["peptideBlosumNum"]
        self.hlaBlosumNum = config["hlaBlosumNum"]
        self.config = config
    def __len__(self):
        return len(self.labelList)
        # return 10

    def __getitem__(self, idx):

        peptide = self.peptideList[idx]
        hla = self.HLAList[idx]
        label = self.labelList[idx]
        
        peptideESM = self.embedder.esmEmbedding(peptide)
        peptideAtchley = self.embedder.atchleyEmbedding(peptide)
        peptideBlosum = self.embedder.blosumEmbedding(peptide, self.peptideBlosumNum)

        hlaESM = self.embedder.esmEmbedding(hla)
        hlaAtchley = self.embedder.atchleyEmbedding(hla)
        hlaBlosum = self.embedder.blosumEmbedding(hla, self.hlaBlosumNum)
        
        context = {
            "label": label,
            "peptide": {
                "ESMc": torch.tensor(peptideESM, dtype=torch.float32),
                "Atchley": torch.tensor(peptideAtchley, dtype=torch.float32),
                "Blosum": torch.tensor(peptideBlosum, dtype=torch.float32),
                "length": len(peptide)
            },
            "hla": {
                "ESMc": torch.tensor(hlaESM, dtype=torch.float32),
                "Atchley": torch.tensor(hlaAtchley, dtype=torch.float32),
                "Blosum": torch.tensor(hlaBlosum, dtype=torch.float32),
                "length": len(hla)
            }
        }
        return context
    

class PTCRDataset(Dataset):
    def __init__(self, csv_path, config):
        self.data = pd.read_csv(csv_path)
        self.peptideList = self.data.peptide.tolist()
        self.tcrList = self.data.tcr.tolist()
        self.labelList = self.data.label.tolist()

        self.embedder = Embedder(
            seqMaxLength=config["seqMaxLength"], 
            esmcEmbeddingDictPath=config["esmcEmbeddingDictPath"],
            atchleyFactorPath=config["atchleyFactorPath"],
            esmcEmbeddingPath=config["esmcEmbeddingPath"])
        
        self.peptideBlosumNum = config["peptideBlosumNum"]
        self.tcrBlosumNum = config["tcrBlosumNum"]
        self.config = config


    def __len__(self):
        return len(self.labelList)
        # return 10

    def __getitem__(self, idx):
        peptide = self.peptideList[idx]
        tcr = self.tcrList[idx]
        label = self.labelList[idx]
        
        peptideESM = self.embedder.esmEmbedding(peptide)
        peptideAtchley = self.embedder.atchleyEmbedding(peptide)
        peptideBlosum = self.embedder.blosumEmbedding(peptide, self.peptideBlosumNum)

        tcrESM = self.embedder.esmEmbedding(tcr)
        tcrAtchley = self.embedder.atchleyEmbedding(tcr)
        tcrBlosum = self.embedder.blosumEmbedding(tcr, self.tcrBlosumNum)
        context = {
                "label": label,
                "tcr": {
                    "ESMc": torch.tensor(tcrESM, dtype=torch.float32),
                    "Atchley": torch.tensor(tcrAtchley, dtype=torch.float32),
                    "Blosum": torch.tensor(tcrBlosum, dtype=torch.float32),
                    "length": len(tcr)
                },
                "peptide": {
                    "ESMc": torch.tensor(peptideESM, dtype=torch.float32),
                    "Atchley": torch.tensor(peptideAtchley, dtype=torch.float32),
                    "Blosum": torch.tensor(peptideBlosum, dtype=torch.float32),
                    "length": len(peptide)
                },
            }
        return context
    

class TCRpHLADataset(Dataset):
    def __init__(self, csv_path, embedder, peptideBlosumNum, tcrBlosumNum, hlaBlosumNum):
        self.embedder = embedder
        self.data = pd.read_csv(csv_path)

        self.peptideList = self.data.peptide.tolist()
        self.tcrList = self.data.tcr.tolist()
        self.hlaList = self.data.hla.tolist()
        self.labelList = self.data.label.tolist()

        self.peptideBlosumNum = peptideBlosumNum
        self.tcrBlosumNum = tcrBlosumNum
        self.hlaBlosumNum = hlaBlosumNum


    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, idx):
        peptide = self.peptideList[idx]
        tcr = self.tcrList[idx]
        hla = self.hlaList[idx]
        label = self.labelList[idx]
        
        peptideESM = self.embedder.esmEmbedding(peptide)
        peptideAtchley = self.embedder.atchleyEmbedding(peptide)
        peptideBlosum = self.embedder.blosumEmbedding(peptide, self.peptideBlosumNum)

        tcrESM = self.embedder.esmEmbedding(tcr)
        tcrAtchley = self.embedder.atchleyEmbedding(tcr)
        tcrBlosum = self.embedder.blosumEmbedding(tcr, self.tcrBlosumNum)

        hlaESM = self.embedder.esmEmbedding(hla)
        hlaAtchley = self.embedder.atchleyEmbedding(hla)
        hlaBlosum = self.embedder.blosumEmbedding(hla, self.hlaBlosumNum)
        
        context = {
                "label": label,
                "tcr": {
                    "ESMc": torch.tensor(tcrESM, dtype=torch.float32),
                    "Atchley": torch.tensor(tcrAtchley, dtype=torch.float32),
                    "Blosum": torch.tensor(tcrBlosum, dtype=torch.float32),
                    "length": len(tcr)
                },
                "peptide": {
                    "ESMc": torch.tensor(peptideESM, dtype=torch.float32),
                    "Atchley": torch.tensor(peptideAtchley, dtype=torch.float32),
                    "Blosum": torch.tensor(peptideBlosum, dtype=torch.float32),
                    "length": len(peptide)
                },
                "hla": {
                    "ESMc": torch.tensor(hlaESM, dtype=torch.float32),
                    "Atchley": torch.tensor(hlaAtchley, dtype=torch.float32),
                    "Blosum": torch.tensor(hlaBlosum, dtype=torch.float32),
                    "length": len(hla)
                }
            }
        # context = (peptideESM, peptideAtchley, peptideBlosum), (tcrESM, tcrAtchley, tcrBlosum), label
        return context



    
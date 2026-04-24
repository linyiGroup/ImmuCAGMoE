import pickle
import os
from pathlib import Path

from tqdm import tqdm

import pandas as pd
import numpy as np
import torch
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.tokenization import EsmSequenceTokenizer



# load embedding dict, if not exist, then return an empty dict
def loadEmbeddingDict(path):
    if not os.path.exists(path):
        return {}
    
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d

# save embedding dict, if not exist, then return an empty dict
def saveEmbeddingDict(embeddingDict, path):
    with open(path, "wb") as f:
        pickle.dump(embeddingDict, f)


# load embedding dict, if not exist, then return an empty dict
def fetchUnembeddedDataPathList(dirsPath):
    filesPath = []
    for dir in dirsPath:
        fileNames = os.listdir(dir)
        filePath = [dir+name for name in fileNames]
        filesPath += filePath
    return filesPath


# get a set, containing all aa sequences of TCR, peptide and MHC pseudo sequence that need to be embedded.
def fetchUnembeddedSequenceSet(dataPathList, squenceColumnName):
    seqSet = set()
    for path in dataPathList:
        if not path.endswith(".csv"):
            continue
        data = pd.read_csv(path)

        for key in squenceColumnName:
            if key in data:
                seqSet.update(data[key].unique().tolist())
    return seqSet



# load ESMc model
def loadESMcModel(path, device):
    device = torch.device(device)
    model = torch.load(path, weights_only=False).to(device)
    return model


# encode one aa seq into id tensor, adding pad to full the id tensor into max seq length
def encodeOneSeq(seq, model, maxSeqLength, device):
    tokenizer = EsmSequenceTokenizer()
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    protein = ESMProtein(sequence=seq)
    protein_tensor = model.encode(protein).to(device)
    
    protein_tensor.sequence = torch.cat([
            protein_tensor.sequence[:-1],    # remove the eos_id
            torch.tensor([pad_id] * (maxSeqLength - len(seq))).to(device),  # full the seq into maxlength using the pad_id 
            torch.tensor([eos_id]).to(device)  # add the eos_id
        ]).long()
    return protein_tensor

# embed one seq of id tensor into feature vector
# return the embedding vector with removed the cls and eos token.
def embedOneSeq(model, protein_tensor):
    logits_output = model.logits(protein_tensor, LogitsConfig(return_embeddings=True))
    embedding = logits_output.embeddings[0, 1:-1]  # remove the cls and eos embedding

    return embedding


# embed all seqs, saving the embeddings
def saveEmbedding(model, seqSet, embeddingDict, maxSeqLength, embeddingDirPath, maxFilesPerDir, device, errorLogPath=""):

    if os.path.exists(errorLogPath):
        os.remove(errorLogPath)
    
    currentDirNum = 1

    with torch.no_grad():
        for seq in tqdm(seqSet):
            try:
                if seq in embeddingDict:
                    continue
                
                savePath = embeddingDirPath + "%s/"%(currentDirNum)
                folder_path = Path(savePath)
                if not folder_path.exists():
                    folder_path.mkdir(parents=True, exist_ok=True)  
                savePath += "%s.npy" % seq

                protein_tensor = encodeOneSeq(seq, model, maxSeqLength, device)

                embedding = embedOneSeq(model, protein_tensor)
                embedding = embedding.cpu().numpy()

                np.save(savePath, embedding)                
                embeddingDict[seq] = "/".join(savePath.split('/')[-2:])
                

            except Exception as e:
                msg = ""
                msg += "error: %s \n" % e
                msg += "seq: %s \n" %seq
                msg += "protein_tensor.sequence: %s \n" % protein_tensor.sequence
                msg += "save path: %s \n\n" % savePath

                print(msg)
                with open(errorLogPath, "a") as f:
                    f.write(msg)

            savePath = embeddingDirPath + "%s/"%(currentDirNum)
            if len(os.listdir(savePath)) >= maxFilesPerDir:
                currentDirNum += 1
                savePath = embeddingDirPath + "%s/"%(currentDirNum)

    return embeddingDict
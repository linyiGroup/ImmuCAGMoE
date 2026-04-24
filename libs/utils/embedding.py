
import pandas as pd
import numpy as np
import pickle
import blosum
import torch


class Embedder(object):
    def __init__(self, seqMaxLength, esmcEmbeddingDictPath, atchleyFactorPath, esmcEmbeddingPath):
        self.seqMaxLength = seqMaxLength
        self.atchleyFactor = pd.read_csv(atchleyFactorPath, index_col=0)
        self.esmcEmbeddingPath = esmcEmbeddingPath
        self.esmcEmbeddingDict = None
        with open(esmcEmbeddingDictPath, "rb") as f:
            self.esmcEmbeddingDict = pickle.load(f)
        
        self.bolum = {}

    def esmEmbedding(self, aaSeq):

        embeddingName = self.esmcEmbeddingDict[aaSeq]
        embedding = np.load(self.esmcEmbeddingPath + embeddingName)
        return embedding

    def atchleyEmbedding(self, aaSeq):
        embedding = [self.atchleyFactor.loc[a].tolist() for a in aaSeq]
        embedding += [[0]*len(embedding[0])] * (self.seqMaxLength - len(embedding))
        embedding = np.array(embedding)
        return embedding

    def blosumEmbedding(self, aaSeq, num):
        if num not in self.bolum:
            self.bolum[num] = blosum.BLOSUM(num)
        bm = self.bolum[num]
        embedding = [list(bm[a].values()) for a in aaSeq]
        embedding += [list(bm["*"].values())] * (self.seqMaxLength - len(embedding))
        embedding = np.array(embedding)
        return embedding




def createCrossAttentionMask(valid_len_a, valid_len_b, len_a, len_b, batch_size):

    mask_a = torch.arange(len_a, device=valid_len_a.device).unsqueeze(0).repeat(batch_size, 1)
    mask_a = mask_a >= valid_len_a.unsqueeze(1)  # (batch_size, len_a)

    mask_b = torch.arange(len_b, device=valid_len_b.device).unsqueeze(0).repeat(batch_size, 1)
    mask_b = mask_b >= valid_len_b.unsqueeze(1)  # (batch_size, len_b)

    attn_mask = mask_a.unsqueeze(2) | mask_b.unsqueeze(1)
    
    return attn_mask


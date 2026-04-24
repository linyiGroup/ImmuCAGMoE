import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append("../") 
from libs.utils.embedding import createCrossAttentionMask



class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config["feat_dim"],
            num_heads=config["num_heads"],
            kdim=config["feat_dim"],
            vdim=config["feat_dim"],
            batch_first=True  
        )

    def forward(self, query, key, value, attn_mask=None):
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool:
                attn_mask = attn_mask == 1 
            batch_size = query.size(0)
            num_heads = self.multihead_attn.num_heads
            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = attn_mask.repeat(1, num_heads, 1, 1)
            attn_mask = attn_mask.view(batch_size * num_heads, attn_mask.size(2), attn_mask.size(3))
            attn_mask = attn_mask.to(dtype=query.dtype) * (-1e9)
        # attn_weights: [batchsize,  seq_len, seq_len]
        output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask  
        )
        return output, attn_weights.detach().cpu()
    



class Adaptor(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )
    def forward(self, x):
        return self.mlp(x)
    
class MultipleEmbeddingFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(config["feat_dim"])

        self.config = config
        self.proj_esm = nn.Linear(config["dim_esm"], config["feat_dim"])  
        self.proj_atchley = nn.Linear(config["dim_atchley"], config["feat_dim"]) 
        self.proj_blosum = nn.Linear(config["dim_blosum"], config["feat_dim"]) 

        self.adaptor_A = Adaptor(config["feat_dim"]) 
        self.adaptor_B = Adaptor(config["feat_dim"]) 
        self.adaptor_C = Adaptor(config["feat_dim"]) 

        self.gate = nn.Linear(config["feat_dim"], 3 * config["seqMaxLength"]) 


    def forward(self, esm, atchley, blosum, mask=None):
        A = self.norm(self.proj_esm(esm))
        B = self.norm(self.proj_atchley(atchley))
        C = self.norm(self.proj_blosum(blosum))

        A = self.relu(self.adaptor_A(B) + self.adaptor_A(C) + A)
        B = self.relu(self.adaptor_B(A) + self.adaptor_B(C) + B)
        C = self.relu(self.adaptor_C(A) + self.adaptor_C(B) + C)

        flattern = A + B + C
        weights = self.gate(flattern.mean(dim=1))  # 形状：(batch_size, 3*seqMaxLength)
        weights_A, weights_B, weights_C = torch.split(weights, self.config["seqMaxLength"], dim=1)
    
        weights_ABC = torch.stack([weights_A, weights_B, weights_C], dim=1)
        weights_ABC = torch.softmax(weights_ABC.transpose(1, 2), dim=2).transpose(1, 2)  
        weights_A, weights_B, weights_C = torch.unbind(weights_ABC, dim=1)

        fused = weights_A.unsqueeze(2) * A + weights_B.unsqueeze(2) * B + weights_C.unsqueeze(2) * C
        return fused


class CrossAttentionExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = CrossAttention(config)
        self.relu = nn.ReLU()

    def forward(self, q, kv, attn_mask):
        f, att = self.layer(q, kv, kv, attn_mask)
        f = self.relu(f)
        return f, att




class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config["feat_dim"], config["feat_dim"]),
            nn.GELU(),
            nn.Linear(config["feat_dim"], config["feat_dim"])
        )

    def forward(self, x):
        return self.mlp(x)

class CAMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_dim = config["feat_dim"]
        self.output_dim = config["feat_dim"]
        self.n_experts = config["n_experts"]
        self.top_k = config["top_k"]
        assert self.top_k <= self.n_experts, "top_k must be less than or equal to n_experts"

        self.experts = nn.ModuleList([
            CrossAttention(config)
            for _ in range(self.n_experts)
        ])

        self.router = nn.Linear(config["feat_dim"], self.n_experts) 


    def forward(self, query, key, value, attn_mask=None):
        x = query
        batch_size, seq_len, _ = x.shape
        global_feat = x.mean(dim=1)  # (batch_size, input_dim)

        raw_weights = self.router(global_feat)  # (batch_size, n_experts)
        top_k_weights, top_k_indices = torch.topk(raw_weights, self.top_k, dim=1)  # (batch_size, top_k)
        top_k_weights = torch.softmax(top_k_weights, dim=1) 
        fused_out = torch.zeros(batch_size, seq_len, self.output_dim, device=x.device)
        att_scores = []
        for k in range(self.top_k):
            expert_ids = top_k_indices[:, k]  # (batch_size,)
            weights = top_k_weights[:, k].unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1)：广播用
            for expert_id in torch.unique(expert_ids):
                sample_mask = (expert_ids == expert_id)  # (batch_size,)
                if not sample_mask.any():
                    continue 
                batch_q = x[sample_mask]  # (num_samples, seq_len, input_dim)
                batch_k = key[sample_mask]
                batch_v = value[sample_mask]
                batch_mask = attn_mask[sample_mask]
                batch_weights = weights[sample_mask]  # (num_samples, 1, 1)
                # attn_weights: [batchsize,  seq_len, seq_len]
                expert_out, attn_weights = self.experts[expert_id](batch_q, batch_k, batch_v, batch_mask)  # (num_samples, seq_len, output_dim)
                fused_out[sample_mask] += expert_out * batch_weights
                att_scores.append(attn_weights)
        # att_scores: [top_k, batchsize,  seq_len, seq_len]
        return fused_out, att_scores, raw_weights 
    
class PHLAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.relu = nn.ReLU()

        self.encoder_P = MultipleEmbeddingFusion(config)
        self.encoder_H = MultipleEmbeddingFusion(config)

        self.CAMoE = CAMoE(self.config)
        self.cls = nn.Linear(self.config["feat_dim"] * self.config["seqMaxLength"], 2)


    def forward(self, inputDict, mask=None):

        mask_p = createCrossAttentionMask(
                valid_len_a=inputDict["peptide"]["length"],
                valid_len_b=inputDict["peptide"]["length"],
                len_a=self.config["seqMaxLength"],
                len_b=self.config["seqMaxLength"],
                batch_size=inputDict["peptide"]["length"].size(0)
            )
        mask_h = createCrossAttentionMask(
                valid_len_a=inputDict["hla"]["length"],
                valid_len_b=inputDict["hla"]["length"],
                len_a=self.config["seqMaxLength"],
                len_b=self.config["seqMaxLength"],
                batch_size=inputDict["hla"]["length"].size(0)
            )
        # batch_maxSeqLength * feat_dim
        f_p = self.relu(self.encoder_P(
                esm=inputDict["peptide"]["ESMc"], 
                atchley=inputDict["peptide"]["Atchley"], 
                blosum=inputDict["peptide"]["Blosum"],
                mask=mask_p
            ))
        # batch_maxSeqLength * feat_dim
        f_h = self.relu(self.encoder_H(
                esm=inputDict["hla"]["ESMc"], 
                atchley=inputDict["hla"]["Atchley"], 
                blosum=inputDict["hla"]["Blosum"],
                mask=mask_h
            ))
        mask_ph = createCrossAttentionMask(
            valid_len_a=inputDict["peptide"]["length"],
            valid_len_b=inputDict["hla"]["length"],
            len_a=self.config["seqMaxLength"],
            len_b=self.config["seqMaxLength"],
            batch_size=inputDict["peptide"]["length"].size(0)
        )

        f_experts, att_scores, raw_weights = self.CAMoE(f_p, f_h, f_h, mask_ph)

        prediction = self.cls(self.relu(f_experts.reshape([f_experts.size(0), -1])))

        return prediction, att_scores, raw_weights




class PTCRModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.relu = nn.ReLU()

        self.encoder_P = MultipleEmbeddingFusion(config)
        self.encoder_T = MultipleEmbeddingFusion(config)

        self.CAMoE = CAMoE(self.config)
        self.cls = nn.Linear(self.config["feat_dim"] * self.config["seqMaxLength"], 2)


    def forward(self, inputDict):

        mask_p = createCrossAttentionMask(
                valid_len_a=inputDict["peptide"]["length"],
                valid_len_b=inputDict["peptide"]["length"],
                len_a=self.config["seqMaxLength"],
                len_b=self.config["seqMaxLength"],
                batch_size=inputDict["peptide"]["length"].size(0)
            )
        mask_t = createCrossAttentionMask(
                valid_len_a=inputDict["tcr"]["length"],
                valid_len_b=inputDict["tcr"]["length"],
                len_a=self.config["seqMaxLength"],
                len_b=self.config["seqMaxLength"],
                batch_size=inputDict["tcr"]["length"].size(0)
            )
        # batch_maxSeqLength * feat_dim
        f_p = self.relu(self.encoder_P(
                esm=inputDict["peptide"]["ESMc"], 
                atchley=inputDict["peptide"]["Atchley"], 
                blosum=inputDict["peptide"]["Blosum"],
                mask=mask_p
            ))
        # batch_maxSeqLength * feat_dim
        f_t = self.relu(self.encoder_T(
                esm=inputDict["tcr"]["ESMc"], 
                atchley=inputDict["tcr"]["Atchley"], 
                blosum=inputDict["tcr"]["Blosum"],
                mask=mask_t
            ))

        mask_pt = createCrossAttentionMask(
                valid_len_a=inputDict["peptide"]["length"],
                valid_len_b=inputDict["tcr"]["length"],
                len_a=self.config["seqMaxLength"],
                len_b=self.config["seqMaxLength"],
                batch_size=inputDict["peptide"]["length"].size(0)
            )
        
        f_experts, att_scores, raw_weights = self.CAMoE(f_p, f_t, f_t, mask_pt)

        prediction = self.cls(self.relu(f_experts.reshape([f_experts.size(0), -1])))

        return prediction, att_scores, raw_weights
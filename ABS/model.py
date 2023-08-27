from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class CondenserLTR(nn.Module):
    def __init__(self, q_enc: PreTrainedModel, p_enc: PreTrainedModel):
        super().__init__()
        self.q_enc = q_enc
        self.p_enc = p_enc

    def encode_query(self, query):
        q_out = self.q_enc(**query, return_dict=True, output_hidden_states=True)
        q_hidden = q_out.hidden_states
        q_reps = (q_hidden[0][:, 0] + q_hidden[-1][:, 0]) / 2
        return q_reps

    def encode_passage(self, passage):
        p_out = self.p_enc(**passage, return_dict=True, output_hidden_states=True)
        p_hidden = p_out.hidden_states
        p_reps = (p_hidden[0][:, 0] + p_hidden[-1][:, 0]) / 2
        return p_reps

    def forward(self, query: Tensor, passage: Tensor, labels: Tensor = None, scores_only=False, encode_only=False):
        # Encode queries and passages
        q_reps = self.encode_query(query)
        p_reps = self.encode_passage(passage)
        if encode_only:
            return {"q_reps": q_reps, "p_reps": p_reps}

        # Contrastive loss
        batch_size = q_reps.size(0)
        psg_per_qry = int(p_reps.size(0) / q_reps.size(0))
        q_idx_map = sum(map(lambda x: [x] * psg_per_qry, range(batch_size)), [])
        scores = q_reps[q_idx_map] * p_reps
        scores = torch.sum(scores, dim=1).view(batch_size, -1)
        if scores_only:
            return scores
        loss = F.cross_entropy(scores, labels)
        # hidden loss is a hack to prevent trainer to filter it out
        return DenseOutput(loss=loss, scores=scores, q_reps=q_reps, p_reps=p_reps)

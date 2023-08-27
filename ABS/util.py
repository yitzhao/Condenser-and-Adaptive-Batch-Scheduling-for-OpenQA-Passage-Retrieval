import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import ndcg_score
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import DataCollatorWithPadding, TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.file_utils import ModelOutput


class ABSCallBack(TrainerCallback):
    """
    TrainerCallback for Adaptive Batch Sampler
    """
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        kwargs['train_dataloader'].batch_sampler.save_checkpoint(checkpoint_dir)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step != state.max_steps:
            kwargs['train_dataloader'].batch_sampler.reset()

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        step_per_epoch = state.max_steps // args.num_train_epochs
        num_step_re_encode = step_per_epoch // 3
        if state.global_step % num_step_re_encode == 0 and state.global_step != state.max_steps:
            kwargs['train_dataloader'].batch_sampler.re_encode()

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()


class DistABSCollactor(DataCollatorWithPadding):
    """
    DataCollator For Adaptive Batch Sampler.
    This sampler must be used together with the provided Adaptive Batch Sampler and DenseTrainer.

    Input: List[Dict{"query":str, "positives" str, "labels": List[int]}]
    Output: Dict{"query": tensor[batch_size, sequence_length],
            "passage": tensor[batch_size*passage_per_query, sequence_length],
            "labels": tensor[batch_size, sequence_length]}.
    """
    q_max_len: int = 32
    p_max_len: int = 128

    def __init__(self, **kwargs):
        super(DistABSCollactor, self).__init__(**kwargs)

    def __call__(self, batch):
        rand = random.Random()
        labels = torch.zeros(len(batch), len(batch), dtype=torch.float32)
        queries = [x['query'] for x in batch]
        passages = []
        for i in range(len(batch)):
            qid = batch[i]['id']
            psg = [batch[i]['positives']]
            for j in range(len(batch)):
                pid = batch[j]['id']
                if pid != qid:
                    psg.append(batch[j]['positives'])
            true_idx = rand.randint(0, len(batch) - 1)
            psg[0], psg[true_idx] = psg[true_idx], psg[0]
            passages.extend(psg)
            labels[i, true_idx] = 1

        queries = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.q_max_len,
            padding=self.padding,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        passages = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.p_max_len,
            padding=self.padding,
            return_token_type_ids=False,
            return_tensors="pt",
        )
        queries = {"input_ids": queries.input_ids, "attention_mask": queries.attention_mask}
        passages = {"input_ids": passages.input_ids, "attention_mask": passages.attention_mask}
        passages['input_ids'] = passages['input_ids'].view(len(batch), len(batch), -1)
        passages['attention_mask'] = passages['attention_mask'].view(len(batch), len(batch), -1)
        return {'query': queries, 'passage': passages, 'labels': labels}


class ABSCollactor(DataCollatorWithPadding):
    """
    DataCollator For Adaptive Batch Sampler.
    This sampler must be used together with the provided Adaptive Batch Sampler and DenseTrainer.

    Input: List[Dict{"query":str, "positives" str, "labels": List[int]}]
    Output: Dict{"query": tensor[batch_size, sequence_length],
            "passage": tensor[batch_size*passage_per_query, sequence_length],
            "labels": tensor[batch_size, sequence_length]}.
    """
    q_max_len: int = 32
    p_max_len: int = 128

    def __init__(self, **kwargs):
        super(ABSCollactor, self).__init__(**kwargs)

    def __call__(self, batch):
        rand = random.Random()
        labels = torch.zeros(len(batch), len(batch), dtype=torch.float32)
        queries = [x['query'] for x in batch]
        passages = []
        for i in range(len(batch)):
            qid = batch[i]['id']
            psg = [batch[i]['positives']]
            for j in range(len(batch)):
                pid = batch[j]['id']
                if pid != qid:
                    psg.append(batch[j]['positives'])
            true_idx = rand.randint(0, len(batch) - 1)
            psg[0], psg[true_idx] = psg[true_idx], psg[0]
            passages.extend(psg)
            labels[i, true_idx] = 1

        queries = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.q_max_len,
            padding=self.padding,
            return_tensors="pt",
        )
        passages = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.p_max_len,
            padding=self.padding,
            return_tensors="pt",
        )
        return {'query': queries, 'passage': passages, 'labels': labels}


class EvalCollactor(DataCollatorWithPadding):
    q_max_len: int = 32
    p_max_len: int = 128

    def __call__(self, feature):
        queries = [x['query'] for x in feature]
        if isinstance(feature[0]['passage'], list):
            passages = [y for x in feature for y in x['passage']]
        else:
            passages = [x['passage'] for x in feature]
        labels = torch.tensor([x['labels'] for x in feature], dtype=torch.float32)
        queries = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.q_max_len,
            padding=True,
            return_tensors="pt",
        )
        passages = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.p_max_len,
            padding=True,
            return_tensors="pt",
        )
        return {'query': queries, 'passage': passages, 'labels': labels}


class DistEvalCollactor(DataCollatorWithPadding):
    q_max_len: int = 32
    p_max_len: int = 128

    def __init__(self, **kwargs):
        super(DistEvalCollactor, self).__init__(**kwargs)

    def __call__(self, feature):
        batch_size = len(feature)
        num_passage = len(feature[0]['passage']) if batch_size > 0 else 0
        queries = [x['query'] for x in feature]
        if isinstance(feature[0]['passage'], list):
            passages = [y for x in feature for y in x['passage']]
        else:
            passages = [x['passage'] for x in feature]
        labels = torch.tensor([x['labels'] for x in feature], dtype=torch.float32)
        queries = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.q_max_len,
            padding=self.padding,
            return_tensors="pt",
        )
        passages = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.p_max_len,
            padding=self.padding,
            return_tensors="pt",
        )
        queries = {"input_ids": queries.input_ids, "attention_mask": queries.attention_mask}
        passages = {"input_ids": passages.input_ids, "attention_mask": passages.attention_mask}
        passages['input_ids'] = passages['input_ids'].view(batch_size, num_passage, -1)
        passages['attention_mask'] = passages['attention_mask'].view(batch_size, num_passage, -1)
        return {'query': queries, 'passage': passages, 'labels': labels}


class EncodeCollactor(DataCollatorWithPadding):
    q_max_len: int = 32
    p_max_len: int = 128

    def __init__(self, **kwargs):
        super(EncodeCollactor, self).__init__(**kwargs)

    def __call__(self, feature):
        queries = [x['query'] for x in feature]
        passages = [x['positives'] for x in feature]
        queries = self.tokenizer(
            queries,
            truncation=True,
            max_length=self.q_max_len,
            padding=self.padding,
            return_tensors="pt",
        )
        passages = self.tokenizer(
            passages,
            truncation=True,
            max_length=self.p_max_len,
            padding=self.padding,
            return_tensors="pt",
        )
        queries = {"input_ids": queries.input_ids, "attention_mask": queries.attention_mask}
        passages = {"input_ids": passages.input_ids, "attention_mask": passages.attention_mask}
        return {'query': queries, 'passage': passages}


class ComputeMetrics:
    """
    Compute MMR and NDCG
    """
    def __call__(self, eval_preds):
        output, labels = eval_preds
        scores = output[2]
        mmr = self.MRR(scores, labels)
        ndcg = self.NDCG(scores, labels)
        return {'mmr': mmr, "ndcg": ndcg}

    @staticmethod
    def MRR(scores, target):
        """
        scores: [batch_size, num_passages]
        """
        probs = softmax(scores, axis=1)
        rank = np.apply_along_axis(rankdata, axis=1, arr=-probs)
        idx = np.argmax(target, axis=1).reshape(-1, 1)
        rank_top_idx = np.take_along_axis(rank, idx, axis=1)  # The rank of the top 1 item of target in prediction
        return np.mean(1 / rank_top_idx)

    @staticmethod
    def Dist(scores):
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        prob = softmax(scores, axis=1)
        prob = -np.sort(-prob, axis=1)
        prob = prob.mean(axis=0)
        prob = (prob * 100).astype(np.int32)
        dist = np.zeros(100)
        start, end = 0, 0
        for i in range(scores.shape[1]):
            end = start + prob[i]
            dist[start:end] = i
            start = end
        return dist[:min(end, len(dist))]

    @staticmethod
    def NDCG(scores, target):
        """
        scores: [batch_size, num_passages]
        """
        probs = softmax(scores, axis=1)
        return ndcg_score(target, probs)

    def compute(self, scores, labels):
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        mmr = self.MRR(scores, labels)
        ndcg = self.NDCG(scores, labels)
        return {'mmr': mmr, "ndcg": ndcg}


@dataclass
class DenseOutput(ModelOutput):
    q_reps: Tensor = None
    p_reps: Tensor = None
    loss: Tensor = None
    scores: Tensor = None


class DevSetPreprocessor:
    def __init__(self):
        self.rand = random.Random()

    def __call__(self, data):
        labels = [1] + [0] * (len(data['passage']) - 1)
        swap_idx = self.rand.randint(0, len(data['passage']) - 1)
        labels[0], labels[swap_idx] = labels[swap_idx], labels[0]
        data['passage'][0], data['passage'][swap_idx] = data['passage'][swap_idx], data['passage'][0]
        data['labels'] = labels
        return data


class TrainSetPreprocessor:
    def __call__(self, data, idx):
        data['id'] = idx
        return data


class SamplerCollector:
    def __init__(self, sampler):
        self.sampler = sampler

    def collect(self):
        mapping = {}
        rand = random.Random()
        dataset = self.sampler.dataset
        for batch in tqdm(self.sampler):
            for qid in batch:
                psg = [qid]
                for pid in batch:
                    if pid != qid:
                        psg.append(pid)
                mapping[qid] = psg

        def map_func(data, idx):
            pids = mapping[idx]
            passage = [data['positives']]
            for _id in pids:
                passage.append(dataset[_id]['positives'])
            labels = [1] + [0] * (len(passage) - 1)
            swap_idx = rand.randint(0, len(passage) - 1)
            labels[0], labels[swap_idx] = labels[swap_idx], labels[0]
            passage[0], passage[swap_idx] = passage[swap_idx], passage[0]
            return {'query': data['query'], 'passage': passage, 'labels': labels}

        return dataset.map(map_func, with_indices=True, num_proc=3, remove_columns=dataset.column_names)

    def refresh(self):
        self.sampler.re_encode()
        self.sampler.reset()


class ABSCollectorCallBack(TrainerCallback):
    """
    TrainerCallback for Adaptive Batch Sampler
    """
    def __init__(self, collector: SamplerCollector):
        self.collector = collector

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step != state.max_steps:
            control.should_save = True
            control.should_training_stop = True

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        torch.cuda.empty_cache()

    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
        self.collector.sampler.save_checkpoint(checkpoint_dir)


class TensorBoardWriter:
    def __init__(self, log_dir):
        self.tb_writer = SummaryWriter(log_dir=log_dir)

    def log_metric(self, step, logs=None, split='train'):
        print(logs)
        if self.tb_writer is not None:
            for k, v in logs.items():
                if isinstance(v, (int, float)):
                    k = f'{split}/{k}'
                    self.tb_writer.add_scalar(k, v, step)
                else:
                    print(
                        "Trainer is attempting to log a value of "
                        f'"{v}" of type {type(v)} for key "{k}" as a scalar. '
                        "This invocation of Tensorboard's writer.add_scalar() "
                        "is incorrect so we dropped this attribute."
                    )
            self.tb_writer.flush()

    def log_parm(self, step, param):
        for k, v in param.items():
            if 'embedding' not in k and 'LayerNorm' not in k:
                self.tb_writer.add_histogram(k, v, step)
        self.tb_writer.flush()

    def on_train_end(self):
        if self.tb_writer:
            self.tb_writer.close()
            self.tb_writer = None


def clean_checkpoints(output_dir, limit):
    ordering_and_checkpoint_path = []
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"checkpoint-*")]
    for path in glob_checkpoints:
        regex_match = re.match(f".*checkpoint-([0-9]+)", path)
        if regex_match is not None and regex_match.groups() is not None:
            ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

    if len(checkpoints_sorted) <= limit:
        return
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        print(f"Deleting older checkpoint [{checkpoint}] due to save_total_limit")
        shutil.rmtree(checkpoint)

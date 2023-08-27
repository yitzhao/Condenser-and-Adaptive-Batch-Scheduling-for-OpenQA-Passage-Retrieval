import argparse

import faiss
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from ABS.model import CondenserLTR


class Processor:
	def __init__(self, query):
		self.query = query

	def __call__(self, data):
		data['query'] = self.query[data['qid']]['text']
		return data


class PassageCollator:
	p_max_len = 128

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, passage):
		passage = [x['text'] for x in passage]
		return self.tokenizer(
			passage,
			truncation=True,
			max_length=self.p_max_len,
			padding=True,
			return_token_type_ids=False,
			return_tensors="pt"
		)


class TestSetCollator:
	q_max_len = 32

	def __init__(self, tokenizer):
		self.tokenizer = tokenizer

	def __call__(self, data):
		pids = torch.tensor([x['pid'] for x in data], dtype=torch.int32)
		qids = torch.tensor([x['qid'] for x in data], dtype=torch.int32)
		query = [x['query'] for x in data]
		query = self.tokenizer(
			query,
			truncation=True,
			max_length=self.q_max_len,
			padding=True,
			return_token_type_ids=False,
			return_tensors="pt",
		)
		return {'query': query, 'pid': pids, 'qid': qids}


class FAISSIndex:
	def __init__(self, dim, device='cpu'):
		self.device = device
		index = faiss.IndexFlatIP(dim)
		if device == 'cuda':
			index = faiss.index_cpu_to_all_gpus(index)
		self.index = index

	def search(self, reps: np.ndarray, top_k: int):
		if not reps.flags.c_contiguous:
			reps = np.asarray(reps, order="C")
		return self.index.search(x=reps, k=top_k)

	def add(self, reps: np.ndarray):
		if not reps.flags.c_contiguous:
			reps = np.asarray(reps, order="C")
		self.index.add(reps)

	def train(self, reps: np.ndarray):
		if not reps.flags.c_contiguous:
			reps = np.asarray(reps, order="C")
		self.index.train(reps)

	def save(self, file):
		index = self.index
		if self.device == 'cuda':
			index = faiss.index_gpu_to_cpu(self.index)
		faiss.write_index(index, file)

	def load(self, file):
		index = faiss.read_index(file)
		if self.device == 'cuda':
			index = faiss.index_cpu_to_all_gpus(index)
		self.index = index
		print(f'load index, ntotal={self.index.ntotal}')

	def __len__(self):
		return self.index.ntotal


def full_ranking(model_path=None):
	MaxMMRRank = 100

	accelerator = Accelerator()

	corpus = load_dataset("Carlisle/msmacro-test-corpus")
	test_set = load_dataset("Carlisle/msmacro-test", split='test_new').map(Processor(corpus['query_new']))

	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco", use_fast=False)
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco', add_pooling_layer=False)
	p_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco', add_pooling_layer=False)
	model = CondenserLTR(q_enc=q_enc, p_enc=p_enc)

	if model_path:
		if accelerator.is_local_main_process:
			print(f"Load Model From {model_path}")
		model_state_dict = torch.load(model_path, map_location='cpu')
		model.load_state_dict(model_state_dict)

	psg_loader = DataLoader(corpus['passage_new'], batch_size=100, num_workers=4,
	                        pin_memory=True, collate_fn=PassageCollator(tokenizer=tokenizer))
	test_loader = DataLoader(test_set, batch_size=100, num_workers=4,
	                         pin_memory=True, collate_fn=TestSetCollator(tokenizer=tokenizer))

	q_enc, p_enc, psg_loader, test_loader = accelerator.prepare(model.q_enc, model.p_enc, psg_loader, test_loader)

	p_index = None
	if accelerator.is_local_main_process:
		p_index = FAISSIndex(dim=768)

	p_enc.eval()
	for batch in tqdm(psg_loader, desc="Encode Passage", disable=not accelerator.is_local_main_process):
		with torch.no_grad():
			p_out = p_enc(**batch, return_dict=True, output_hidden_states=True)
		p_hidden = p_out.hidden_states
		p_reps = (p_hidden[0][:, 0] + p_hidden[-1][:, 0]) / 2
		p_reps = accelerator.gather(p_reps)
		if accelerator.is_local_main_process:
			p_index.add(p_reps.detach().cpu().numpy())

	MMR = 0
	counter = 0
	q_enc.eval()
	for batch in tqdm(test_loader, desc="Compute MMR", disable=not accelerator.is_local_main_process):
		with torch.no_grad():
			q_out = q_enc(**batch['query'], return_dict=True, output_hidden_states=True)
		q_hidden = q_out.hidden_states
		q_reps = (q_hidden[0][:, 0] + q_hidden[-1][:, 0]) / 2
		target_pids = batch['pid']
		q_reps = accelerator.gather(q_reps)
		target_pids = accelerator.gather(target_pids)

		if accelerator.is_local_main_process:
			_, candidate_pids = p_index.search(q_reps.detach().cpu().numpy(), top_k=MaxMMRRank)
			target_pids = target_pids.detach().cpu().numpy()
			counter += target_pids.shape[0]
			for i in range(len(candidate_pids)):
				for j in range(MaxMMRRank):
					if candidate_pids[i][j] in target_pids[i]:
						MMR += 1 / (j + 1)
						break
			print(MMR / counter)

	if accelerator.is_local_main_process:
		MMR = MMR / counter
		print(model_path, MMR)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Full Ranking")
	parser.add_argument("--model", required=True)
	args = parser.parse_args()
	full_ranking(args.model)

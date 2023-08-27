import os
from typing import Union, Dict

import numpy as np
import torch
from datasets import load_from_disk, Dataset
from torch.utils.data import Sampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizerBase


class AdaptiveBatchSampler(Sampler[int]):
	"""
	AdaptiveBatchSampler based on Encode Vector dot-product similarity.
	This sampler must be used together with the provided DenseTrainer.

	Dataset Format Requirement:
	Column: ["id": int, "query": str, "positives": str]

	Sampler output: A list of id. e.g. [1,43,5,23,12,42,53,55]
	"""

	def __init__(self,
	             dataset: Dataset,
	             q_enc: PreTrainedModel,
	             p_enc: PreTrainedModel,
	             tokenizer: PreTrainedTokenizerBase,
	             batch_size: int = 8,
	             drop_last: bool = True,
	             sim_cache: Union[str, Dict[str, torch.Tensor]] = None,
	             resume_from_checkpoint: str = None):
		"""
		:param dataset: dataset to sample
		:param q_enc: model for query encoding
		:param p_enc: model for passage encoding
		:param tokenizer: model tokenizer
		:param sim_cache: cache file containing encode query and passage.
		If None, then re-encode query and passage on initialization
		:param batch_size: number of samples in a batch
		:param resume_from_checkpoint: directory of training checkpoint to resume from
		"""

		super().__init__(data_source=dataset)

		if resume_from_checkpoint:
			self.sim = Similarity(cache=os.path.join(resume_from_checkpoint, "sim.npz"))
			self.U = np.load(os.path.join(resume_from_checkpoint, "U.npy"))
			self.T = [x.tolist() for x in np.load(os.path.join(resume_from_checkpoint, "T.npy"))]
			self.hardness_log = np.loadtxt(os.path.join(resume_from_checkpoint, "hardness.txt")).tolist()
		else:
			if sim_cache:
				self.sim = Similarity(cache=sim_cache)
			else:
				self.sim = Similarity(q_enc=q_enc, p_enc=p_enc, tokenizer=tokenizer, dataset=dataset)
			self.hardness_log = []
			self.U = np.arange(len(dataset), dtype=np.int32)
			self.T = []

		self.dataset = dataset
		self.batch_size = batch_size
		self.drop_last = drop_last
		self.p_enc = p_enc
		self.q_enc = q_enc
		self.tokenizer = tokenizer

		# Find passages that are answers to multiple questions
		dup_p = {}
		passage_set = {}
		for i in range(len(dataset)):
			passage = dataset[i]['positives']
			if passage in passage_set:
				passage_set[passage].add(i)
				if len(passage_set[passage]) > 1:
					dup_p[passage] = passage_set[passage]
			else:
				passage_set[passage] = {i}

		self.dup_set = {}
		for dup_set in dup_p.values():
			for idx in dup_set:
				self.dup_set[idx] = dup_set - {idx}

	@staticmethod
	def get_remove(cache):
		sim_arr = cache.get()
		diff = torch.sum(sim_arr, dim=1) + torch.sum(sim_arr, dim=0)
		min_idx = torch.argmin(diff)
		pair_remove, sub = cache.qid[min_idx], diff[min_idx]
		return pair_remove, sub

	@staticmethod
	def get_add(current_dataset, pair_remove, remain_dataset, cache_b_u=None, cache_u_b=None):
		s1 = cache_u_b.get().sum(axis=1)
		s1 -= cache_u_b.get_passage_sim(pair_remove)
		idx = [cache_u_b.qid2rowid[x] for x in current_dataset]
		s1[idx] = -np.inf

		s2 = cache_b_u.get().sum(axis=0)
		s2 -= cache_b_u.get_query_sim(pair_remove)
		idx = [cache_b_u.pid2colid[x] for x in current_dataset]
		s2[idx] = -np.inf

		diff = s1 + s2
		max_idx = torch.argmax(diff)
		pair_add, add = remain_dataset[max_idx], diff[max_idx]

		return pair_add, add

	def save_checkpoint(self, checkpoint_dir):
		os.makedirs(checkpoint_dir, exist_ok=True)
		np.save(os.path.join(checkpoint_dir, "U.npy"), self.U)
		np.save(os.path.join(checkpoint_dir, "T.npy"), np.vstack(self.T))
		np.savetxt(os.path.join(checkpoint_dir, "hardness.txt"), np.array(self.hardness_log))
		self.sim.save(os.path.join(checkpoint_dir, "sim.npz"))

	def re_encode(self, cache=None):
		self.sim = Similarity(q_enc=self.q_enc, p_enc=self.p_enc, cache=cache,
		                      tokenizer=self.tokenizer, dataset=self.dataset)

	def reset(self):
		self.T = []
		self.U = np.arange(len(self.dataset), dtype=np.int32)

	def __iter__(self):
		yield from self.T

		total = len(self.dataset) // self.batch_size - len(self.T)
		for i in range(total):
			B = np.random.choice(self.U, size=self.batch_size, replace=False)
			cache_B = SimilarityCache(self.sim, self.dup_set, B, B)
			cache_B_U = SimilarityCache(self.sim, self.dup_set, B, self.U)
			cache_U_B = SimilarityCache(self.sim, self.dup_set, self.U, B)
			hardness_B = cache_B.get().sum()
			prev_d_r, prev_d_a = -1, -1
			if self.U.shape[0] > self.batch_size:
				while True:
					d_r, sub = self.get_remove(cache_B)
					d_a, add = self.get_add(B, d_r, self.U, cache_b_u=cache_B_U, cache_u_b=cache_U_B)
					hardness_B_tem = hardness_B - sub + add
					if hardness_B_tem > hardness_B and d_r != prev_d_a and d_a != prev_d_r:
						B = np.setdiff1d(B, d_r)
						B = np.append(B, d_a)
						hardness_B = hardness_B_tem
						cache_B.swap_query(d_r, d_a)
						cache_B.swap_passage(d_r, d_a)
						cache_B_U.swap_query(d_r, d_a)
						cache_U_B.swap_passage(d_r, d_a)
						prev_d_r, prev_d_a = d_r, d_a
					else:
						break
			self.T.append(B.tolist())
			self.U = np.setdiff1d(self.U, B)
			self.hardness_log.append(hardness_B.item())
			yield B.tolist()

	def __len__(self):
		return len(self.dataset) // self.batch_size


class Similarity:
	def __init__(self,
	             q_enc: PreTrainedModel = None,
	             p_enc: PreTrainedModel = None,
	             tokenizer: PreTrainedTokenizerBase = None,
	             dataset: Dataset = None,
	             cache: Union[str, Dict[str, torch.Tensor]] = None):

		self.dataset = dataset
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.q_enc = q_enc.to(self.device) if q_enc else None
		self.p_enc = p_enc.to(self.device) if p_enc else None
		self.tokenizer = tokenizer

		if cache is not None:
			if isinstance(cache, str):
				cache = np.load(cache)
				self.query = torch.from_numpy(cache['query']).to(self.device)
				self.passage = torch.from_numpy(cache['passage']).T.to(self.device)
			else:
				self.query = cache['query']
				self.passage = cache['passage'].T
		else:
			self.query = torch.zeros(len(dataset), 768, dtype=torch.float32).to(self.device)
			self.passage = torch.zeros(768, len(dataset), dtype=torch.float32).to(self.device)
			queries = dataset['query']
			passages = dataset['positives']

			self.q_enc.eval()
			self.p_enc.eval()
			with torch.no_grad():
				step = 200
				for i in tqdm(range(0, len(dataset), step), desc="Encode query and passage"):
					start, end = i, min(i + step, len(dataset))
					q = self.tokenizer(queries[start:end], return_tensors="pt", padding=True,
					                   truncation=True, return_token_type_ids=False, max_length=32).to(self.device)
					p = self.tokenizer(passages[start:end], return_tensors="pt", padding=True,
					                   truncation=True, return_token_type_ids=False, max_length=128).to(self.device)
					q_out = self.q_enc(**q, output_hidden_states=True)
					p_out = self.p_enc(**p, output_hidden_states=True)
					self.query[start:end] = (q_out.hidden_states[0][:, 0] + q_out.hidden_states[-1][:, 0]) / 2
					self.passage[:, start:end] = ((p_out.hidden_states[0][:, 0] + p_out.hidden_states[-1][:, 0]) / 2).T
				torch.cuda.empty_cache()

	def get_sim(self, qid=None, pid=None):
		if qid is not None and pid is not None:
			return self.query[qid] @ self.passage[:, pid]
		elif qid is not None and pid is None:
			return self.query[qid] @ self.passage
		elif qid is None and pid is not None:
			return self.query @ self.passage[:, pid]
		else:
			return None

	def save(self, file: str):
		query = self.query.cpu().detach().numpy()
		passage = self.passage.cpu().detach().numpy().T
		np.savez_compressed(file, query=query, passage=passage)


class SimilarityCache:
	def __init__(self, sim: Similarity, dup_set, qid, pid):
		self.sim = sim
		self.dup_set = dup_set
		self.qid = qid
		self.pid = pid
		self.qid2rowid = {qid[i]: i for i in range(len(qid))}
		self.pid2colid = {pid[i]: i for i in range(len(pid))}
		self.cache = sim.get_sim(qid, pid)

		if len(qid) <= len(pid):
			for _id in self.qid:
				# Set S_ii = 0
				if _id in self.pid2colid:
					self.cache[self.qid2rowid[_id], self.pid2colid[_id]] = 0
				# If a passage is answer to multiple queries, it should not be considered as negative sample
				if _id in self.dup_set:
					for dup_id in self.dup_set[_id]:
						if dup_id in self.pid2colid:
							self.cache[self.qid2rowid[_id], self.pid2colid[dup_id]] = 0
		else:
			for _id in self.pid:
				# Set S_ii = 0
				if _id in self.qid2rowid:
					self.cache[self.qid2rowid[_id], self.pid2colid[_id]] = 0
				# If a passage is answer to multiple queries, it should not be considered as negative sample
				if _id in self.dup_set:
					for dup_id in self.dup_set[_id]:
						if dup_id in self.qid2rowid:
							self.cache[self.qid2rowid[dup_id], self.pid2colid[_id]] = 0

	def get(self):
		return self.cache

	def swap_query(self, old_qid, new_qid):
		new_score = self.sim.get_sim([new_qid], self.pid)
		if new_qid in self.pid2colid:
			new_score[0, self.pid2colid[new_qid]] = 0
		if new_qid in self.dup_set:
			for dup_id in self.dup_set[new_qid]:
				if dup_id in self.pid2colid:
					new_score[0, self.pid2colid[dup_id]] = 0

		row_idx = self.qid2rowid[old_qid]
		self.cache[row_idx] = new_score
		self.qid[row_idx] = new_qid
		del self.qid2rowid[old_qid]
		self.qid2rowid[new_qid] = row_idx

	def swap_passage(self, old_pid, new_pid):
		new_score = self.sim.get_sim(self.qid, [new_pid]).flatten()
		if new_pid in self.qid2rowid:
			new_score[self.qid2rowid[new_pid]] = 0
		if new_pid in self.dup_set:
			for dup_id in self.dup_set[new_pid]:
				if dup_id in self.qid2rowid:
					new_score[self.qid2rowid[dup_id]] = 0

		col_idx = self.pid2colid[old_pid]
		self.cache[:, col_idx] = new_score
		self.pid[col_idx] = new_pid
		del self.pid2colid[old_pid]
		self.pid2colid[new_pid] = col_idx

	def get_query_sim(self, qid):
		return self.cache[self.qid2rowid[qid]]

	def get_passage_sim(self, pid):
		return self.cache[:, self.pid2colid[pid]]


if __name__ == '__main__':
	samples = load_from_disk("Dataset/sample_1000_raw")
	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	p_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	ABS = AdaptiveBatchSampler(dataset=samples, q_enc=q_enc, p_enc=p_enc,
	                           tokenizer=tokenizer, sim_cache="Dataset/sim_cache_1000.npz")

	it = iter(ABS)
	for batch in range(len(ABS) - 50):
		print(next(it))

	ABS.save_checkpoint("abs_output/checkpoint_test")

	ABS2 = AdaptiveBatchSampler(dataset=samples, q_enc=q_enc, p_enc=p_enc, tokenizer=tokenizer,
	                            resume_from_checkpoint="abs_output/checkpoint_test")
	for batch in ABS2:
		print(batch)
	ABS2.reset()

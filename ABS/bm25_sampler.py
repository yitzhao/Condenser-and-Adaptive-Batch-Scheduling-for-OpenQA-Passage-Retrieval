import math
import os
from collections import Counter

import numpy as np
import torch
from datasets import load_from_disk, Dataset
from torch.utils.data import Sampler
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


class AdaptiveBatchSampler(Sampler[int]):
	"""
	AdaptiveBatchSampler based on BM25 similarity.
	This sampler must be used together with the provided DenseTrainer.

	Dataset Format Requirement:
	Column: ["id": int, "query": str, "positives": str]

	Sampler output: A list of id. e.g. [1,43,5,23,12,42,53,55]
	"""

	def __init__(self,
	             dataset: Dataset,
	             tokenizer: PreTrainedTokenizerBase,
	             batch_size: int = 8,
	             resume_from_checkpoint: str = None):
		"""
		:param dataset: dataset to sample
		:param tokenizer: tokenizer that used in the model
		:param batch_size: number of samples pair in a batch
		:param resume_from_checkpoint: directory of training checkpoint to resume from
		"""

		super().__init__(data_source=dataset)

		self.dataset = dataset
		self.batch_size = batch_size
		self.tokenizer = tokenizer

		self.sim = BM25(dataset['positives'], tokenizer=tokenizer)
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.query = self.tokenizer(dataset['query'], return_tensors="pt", padding=True, max_length=32,
		                            return_attention_mask=False, add_special_tokens=False, truncation=True,
		                            return_token_type_ids=False).input_ids.to(self.device)

		if resume_from_checkpoint:
			self.U = np.load(os.path.join(resume_from_checkpoint, "U.npy"))
			self.T = [x.tolist() for x in np.load(os.path.join(resume_from_checkpoint, "T.npy"))]
			self.hardness_log = np.loadtxt(os.path.join(resume_from_checkpoint, "hardness.txt")).tolist()
		else:
			self.hardness_log = []
			self.U = np.arange(len(dataset), dtype=np.int32)
			self.T = []

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

	def re_encode(self):
		pass

	def reset(self):
		self.U = np.arange(len(self.dataset), dtype=np.int32)
		self.T = []

	def __iter__(self):
		yield from self.T

		total = len(self.dataset) // self.batch_size - len(self.T)
		for i in range(total):
			B = np.random.choice(self.U, size=self.batch_size, replace=False)
			cache_B = SimilarityCache(self.sim, self.query, self.dup_set, B, B)
			cache_B_U = SimilarityCache(self.sim, self.query, self.dup_set, B, self.U)
			cache_U_B = SimilarityCache(self.sim, self.query, self.dup_set, self.U, B)
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


class BM25(object):
	def __init__(self, corpus, tokenizer: PreTrainedTokenizerBase, k1=1.5, b=0.75, epsilon=0.25):
		self.k1 = k1
		self.b = b
		self.epsilon = epsilon
		self.tokenizer = tokenizer
		self.vocab_size = tokenizer.vocab_size
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

		self.corpus_size = len(corpus)
		self.avgdl = 0

		self.doc_freqs = np.zeros((len(corpus), self.vocab_size), dtype=np.uint8)
		self.idf = np.zeros(self.vocab_size)
		self.doc_len = np.zeros(len(corpus))

		self._initialize(corpus)
		self.doc_freqs = torch.tensor(self.doc_freqs, dtype=torch.uint8, device=self.device)
		self.idf = torch.tensor(self.idf, device=self.device)
		self.denominator_constant = torch.tensor(self.denominator_constant, device=self.device)
		self.denominator_constant2 = self.denominator_constant.reshape(-1, 1, 1)
		self.numerator_constant = torch.tensor(self.k1 + 1, dtype=torch.float16)

	def _initialize(self, corpus):
		"""Calculates frequencies of terms in documents and in corpus. Also computes inverse document frequencies."""
		nd = {}  # word -> number of documents with word
		num_doc = 0
		for i in tqdm(range(len(corpus)), desc="Count Frequency"):
			document = self.encode(corpus[i], return_tensors='np')
			self.doc_len[i] = len(document)
			num_doc += len(document)

			frequencies = Counter(document)
			for token_id in frequencies:
				self.doc_freqs[i, token_id] += frequencies[token_id]
				nd[token_id] = nd.get(token_id, 0) + 1

		self.avgdl = float(num_doc) / self.corpus_size
		# collect idf sum to calculate an average idf for epsilon value
		idf_sum = 0
		# collect words with negative idf to set them a special epsilon value.
		# idf can be negative if word is contained in more than half of documents
		negative_idfs = []
		for token_id, freq in tqdm(nd.items(), desc="Count IDF"):
			idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
			self.idf[token_id] = idf
			idf_sum += idf
			if idf < 0:
				negative_idfs.append(token_id)
		self.average_idf = float(idf_sum) / len(nd)

		if self.average_idf < 0:
			print(
				'Average inverse document frequency is less than zero. Your corpus of {} documents'
				' is either too small or it does not originate from natural text. BM25 may produce'
				' unintuitive results.'.format(self.corpus_size)
			)

		eps = self.epsilon * self.average_idf
		for token_id in negative_idfs:
			self.idf[token_id] = eps

		self.denominator_constant = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

	def get_sim(self, queries, indexes, encoded=False):
		if not encoded:
			queries = self.batch_encode(queries).to(self.device)
		denominator_constant = self.denominator_constant2[indexes]

		df = torch.zeros(len(indexes), queries.size(0), queries.size(1), device=self.device)
		idf = self.idf[queries]

		if len(queries) <= len(indexes):
			for i in range(len(queries)):
				df[:, i, :] = self.doc_freqs[:, queries[i]][indexes]
		else:
			for i in range(len(indexes)):
				df[i] = self.doc_freqs[indexes[i]][queries]

		n = df * idf * self.numerator_constant
		d = df + denominator_constant
		score = torch.sum(n / d, dim=-1).T
		return score

	def encode(self, text, return_tensors="pt"):
		return self.tokenizer.encode(text, return_tensors=return_tensors, add_special_tokens=False,
		                             padding=False, truncation=True).flatten()

	def batch_encode(self, text, return_tensors="pt"):
		return self.tokenizer(text, return_tensors=return_tensors, padding=True, return_attention_mask=False,
		                      add_special_tokens=False, return_token_type_ids=False).input_ids


class SimilarityCache:
	def __init__(self, sim: BM25, query, dup_set, qid, pid):
		self.sim = sim
		self.query = query
		self.dup_set = dup_set
		self.qid = qid
		self.pid = pid
		self.qid2rowid = {qid[i]: i for i in range(len(qid))}
		self.pid2colid = {pid[i]: i for i in range(len(pid))}
		self.cache = sim.get_sim(query[qid], pid, encoded=True)

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
		new_score = self.sim.get_sim(self.query[new_qid].reshape(1, -1), self.pid, encoded=True)
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
		new_score = self.sim.get_sim(self.query[self.qid], [new_pid], encoded=True).flatten()
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
	# Test Code
	samples = load_from_disk("Dataset/sample_1000_raw")
	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	ABS = AdaptiveBatchSampler(dataset=samples, tokenizer=tokenizer)

	it = iter(ABS)
	for batch in range(len(ABS) - 50):
		print(next(it))

	ABS.save_checkpoint("abs_output/checkpoint_test")

	ABS2 = AdaptiveBatchSampler(dataset=samples, tokenizer=tokenizer,
	                            resume_from_checkpoint="abs_output/checkpoint_test")
	for batch in ABS2:
		print(batch)
	ABS2.reset()

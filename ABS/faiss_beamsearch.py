import os.path
from typing import Union, Dict

import faiss
import numpy as np
import torch
from datasets import load_from_disk, Dataset
from torch.utils.data import Sampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PreTrainedTokenizerBase


class AdaptiveBatchSampler(Sampler[int]):
	"""
	AdaptiveBatchSampler based on dot-product similarity.
	Faiss and Beam Search are applied to this implementation.
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
	             beam_search_range: int = 3,
	             batch_size: int = 8,
	             sim_cache: Union[str, Dict[str, torch.Tensor]] = None,
	             resume_from_checkpoint: str = None):

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
		self.q_enc = q_enc
		self.p_enc = p_enc
		self.tokenizer = tokenizer
		self.batch_size = batch_size
		self.beam_search_range = beam_search_range

		self.q_reps = self.sim.get_query().detach().cpu().numpy()
		self.p_reps = self.sim.get_passage().detach().cpu().numpy()
		self.qry_idx = FAISSIndex(self.q_reps)
		self.psg_idx = FAISSIndex(self.p_reps)

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
	def get_remove(cache, beam_search_index):
		sim_arr = cache.get()
		diff = torch.sum(sim_arr, dim=1) + torch.sum(sim_arr, dim=0)
		top_k = torch.topk(diff, k=beam_search_index)
		return cache.qid[top_k.indices.cpu()], top_k.values.cpu().numpy()

	@staticmethod
	def get_add(self, batch_size, current_dataset, pair_remove, remain_dataset, topk=100,
	            beam_search_range=3):
		remain_dataset = np.setdiff1d(remain_dataset, current_dataset)
		if len(remain_dataset) <= 2 * batch_size * topk:
			candidates = remain_dataset
		else:
			# FAISS
			_, candidates_1 = self.psg_idx.search(self.q_reps[current_dataset], top_k=topk)
			_, candidates_2 = self.qry_idx.search(self.p_reps[current_dataset], top_k=topk)
			candidates = np.union1d(candidates_1.flatten(), candidates_2.flatten())
			candidates = np.intersect1d(candidates, remain_dataset)

		cache_b_u = SimilarityCache(self.sim, self.dup_set, current_dataset, candidates)
		cache_u_b = SimilarityCache(self.sim, self.dup_set, candidates, current_dataset)

		s1 = cache_u_b.get().sum(axis=1) - cache_u_b.get_passage_sim(pair_remove)
		s2 = cache_b_u.get().sum(axis=0) - cache_b_u.get_query_sim(pair_remove)

		if len(s1) == 0:
			return [], []

		top_k = torch.topk(s1 + s2, k=min(beam_search_range, len(s1)))
		pair_add, add = candidates[top_k.indices.cpu()], top_k.values.cpu().numpy()
		if not isinstance(pair_add, np.ndarray):
			pair_add = np.array([pair_add])
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
		self.q_reps = self.sim.get_query().detach().cpu().numpy()
		self.p_reps = self.sim.get_passage().detach().cpu().numpy()
		self.qry_idx = FAISSIndex(self.q_reps)
		self.psg_idx = FAISSIndex(self.p_reps)

	def reset(self):
		self.U = np.arange(len(self.dataset), dtype=np.int32)
		self.T = []

	def __len__(self):
		return len(self.dataset) // self.batch_size

	def __iter__(self):
		yield from self.T

		total = len(self.dataset) // self.batch_size - len(self.T)
		for _ in range(total):
			B = np.random.choice(self.U, size=self.batch_size, replace=False)
			cache_B = SimilarityCache(self.sim, self.dup_set, B, B)
			hardness_B = cache_B.get().sum()
			prev_d_r, prev_d_a = -1, -1
			if self.U.shape[0] > self.batch_size:
				while True:
					hardness_B_tem = -np.inf
					d_r, d_a = None, None
					d_r_list, sub_list = self.get_remove(cache_B, self.beam_search_range)
					for i in range(len(d_r_list)):
						d_a_list, add_list = self.get_add(self, self.batch_size, B, d_r_list[i],
						                                  self.U, 100, self.beam_search_range)
						for j in range(len(d_a_list)):
							if hardness_B - sub_list[i] + add_list[j] > hardness_B_tem:
								hardness_B_tem = hardness_B - sub_list[i] + add_list[j]
								d_r = d_r_list[i]
								d_a = d_a_list[j]
					if hardness_B_tem > hardness_B and d_r != prev_d_a and d_a != prev_d_r:
						B = np.setdiff1d(B, d_r)
						B = np.append(B, d_a)
						hardness_B = hardness_B_tem
						cache_B = SimilarityCache(self.sim, self.dup_set, B, B)
						prev_d_r, prev_d_a = d_r, d_a
					else:
						break

			self.U = np.setdiff1d(self.U, B)
			self.T.append(B.tolist())
			self.hardness_log.append(hardness_B.item())
			yield B.tolist()


class Similarity:
	def __init__(self,
	             q_enc: PreTrainedModel = None,
	             p_enc: PreTrainedModel = None,
	             tokenizer=None,
	             dataset=None,
	             cache: Union[str, Dict[str, torch.Tensor]] = None):

		self.dataset = dataset
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.q_enc = q_enc.to(self.device) if q_enc else None
		self.p_enc = p_enc.to(self.device) if p_enc else None
		self.tokenizer = tokenizer

		if cache is not None:
			if isinstance(cache, str):
				cache = np.load(cache)
				self.query = torch.from_numpy(cache['query'])
				self.passage = torch.from_numpy(cache['passage']).T
			else:
				self.query = cache['query']
				self.passage = cache['passage'].T
		else:
			self.query = torch.zeros(len(dataset), 768, dtype=torch.float32)
			self.passage = torch.zeros(768, len(dataset), dtype=torch.float32)
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
					q_rep = (q_out.hidden_states[0][:, 0] + q_out.hidden_states[-1][:, 0]) / 2
					p_rep = ((p_out.hidden_states[0][:, 0] + p_out.hidden_states[-1][:, 0]) / 2).T
					self.query[start:end] = q_rep.cpu()
					self.passage[:, start:end] = p_rep.cpu()
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

	def get_query(self):
		return self.query

	def get_passage(self):
		return self.passage.T


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

	def get_query_sim(self, qid):
		return self.cache[self.qid2rowid[qid]]

	def get_passage_sim(self, pid):
		return self.cache[:, self.pid2colid[pid]]


class FAISSIndex:
	def __init__(self, reps: np.ndarray):
		self.dim = reps.shape[1]
		index = faiss.index_factory(self.dim, "IVF64,Flat,RFlat", faiss.METRIC_INNER_PRODUCT)
		index.nprobe = 64
		self.index = index
		self.train(reps)
		self.add(reps)

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

	def __len__(self):
		return self.index.ntotal


if __name__ == "__main__":
	samples = load_from_disk("../../Dataset/sample_1000_raw")
	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	q_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	p_enc = AutoModel.from_pretrained('Luyu/co-condenser-marco')
	ABS = AdaptiveBatchSampler(dataset=samples, q_enc=q_enc, p_enc=p_enc,
	                           tokenizer=tokenizer, sim_cache="../../Dataset/sim_cache_1000.npz")

	it = iter(ABS)
	for batch in range(len(ABS)):
		print(next(it))

	ABS.save_checkpoint("abs_output/checkpoint_test")

	ABS2 = AdaptiveBatchSampler(dataset=samples, q_enc=q_enc, p_enc=p_enc, tokenizer=tokenizer,
	                            resume_from_checkpoint="abs_output/checkpoint_test")
	for batch in ABS2:
		print(batch)
	ABS2.reset()

import argparse
import os.path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, TrainerState

from ABS.bm25_sampler import AdaptiveBatchSampler as ABSBM25Sampler
from ABS.encode_sampler import AdaptiveBatchSampler as ABSEncodeSampler
from ABS.faiss_beamsearch import AdaptiveBatchSampler as ABSBeamSearchSampler
from ABS.model import CondenserLTR
from ABS.util import DistEvalCollactor, ComputeMetrics, DistABSCollactor, \
	TensorBoardWriter, clean_checkpoints, EncodeCollactor

WEIGHT_NAME = "pytorch_model.bin"
OPTIMIZER_STATE = "optimizer.pt"
LOGGING_DIR = "runs"
TRAINER_STATE_NAME = "trainer_state.json"
SCHEDULER_NAME = "scheduler.pt"

ABSEncode = "abs-encode"
ABSBM25 = "abs-bm25"
ABSBeamSearch = 'abs-beamsearch'
Random = "random"
Sequential = "sequential"


def encode(accelerator, model, dataloader):
	if accelerator.is_local_main_process:
		print("Start Encoding")
	model.eval()
	all_q_reps, all_p_reps = None, None
	for batch in tqdm(dataloader, disable=not accelerator.is_local_main_process):
		with torch.no_grad():
			output = model(**batch, encode_only=True)
		output = accelerator.gather(output)
		all_q_reps = output["q_reps"] if all_q_reps is None else torch.cat([all_q_reps, output["q_reps"]], dim=0)
		all_p_reps = output["p_reps"] if all_p_reps is None else torch.cat([all_p_reps, output["p_reps"]], dim=0)
	return {"query": all_q_reps, "passage": all_p_reps}


def training(config, args):
	state = TrainerState()
	accelerator = Accelerator(fp16=args.fp16, cpu=args.cpu, dispatch_batches=True)
	lr = config["lr"]
	batch_size = config["batch_size"]
	eval_batch_size = config['eval_batch_size']
	logging_step = config['logging_step']
	output_dir = config['output_dir']
	save_total_limit = config['save_total_limit']
	model_path = config['model_path']
	resume_from_checkpoint = config["resume_from_checkpoint"]

	tb_writer = None
	if accelerator.is_local_main_process:
		tb_writer = TensorBoardWriter(log_dir=os.path.join(output_dir, LOGGING_DIR))

	train_set = load_dataset("Carlisle/msmarco-passage-abs", split='train').select(list(range(args.train_size)))
	dev_set = load_dataset("Carlisle/msmarco-passage-non-abs", split='dev').select(list(range(args.dev_size)))
	tokenizer = AutoTokenizer.from_pretrained(args.pretrain, use_fast=False)

	q_enc = AutoModel.from_pretrained(args.pretrain, add_pooling_layer=False)
	p_enc = AutoModel.from_pretrained(args.pretrain, add_pooling_layer=False)
	model = CondenserLTR(q_enc=q_enc, p_enc=p_enc)

	optimizer = AdamW(params=model.parameters(), lr=lr)
	compute_metrics = ComputeMetrics()

	# Load Model If Model Path Is Given
	if model_path and not resume_from_checkpoint:
		if accelerator.is_local_main_process:
			print(f"Load Model From {model_path}")
		model_state_dict = torch.load(os.path.join(model_path, WEIGHT_NAME), map_location='cpu')
		model.load_state_dict(model_state_dict)

	# Resume Model From Checkpoint
	if resume_from_checkpoint:
		if accelerator.is_local_main_process:
			print(f"Resume From {resume_from_checkpoint}")
		model_state_dict = torch.load(os.path.join(resume_from_checkpoint, WEIGHT_NAME), map_location='cpu')
		model.load_state_dict(model_state_dict)

	encode_dataloader = DataLoader(dataset=train_set, batch_size=200, num_workers=4, pin_memory=True,
	                               collate_fn=EncodeCollactor(tokenizer=tokenizer, padding='max_length'))
	model, encode_dataloader = accelerator.prepare(model, encode_dataloader)

	# Choose Sampler
	if args.sampler == ABSEncode:
		sim_cache = encode(accelerator, model, encode_dataloader)
		if accelerator.is_local_main_process:
			print("Use ABSEncode Sampler")
		sampler = ABSEncodeSampler(dataset=train_set, q_enc=q_enc, p_enc=p_enc,
		                           tokenizer=tokenizer, batch_size=batch_size, sim_cache=sim_cache,
		                           resume_from_checkpoint=resume_from_checkpoint)
	elif args.sampler == ABSBeamSearch:
		sim_cache = encode(accelerator, model, encode_dataloader)
		if accelerator.is_local_main_process:
			print("Use ABSBeamSearch Sampler")
		sampler = ABSBeamSearchSampler(dataset=train_set, q_enc=q_enc, p_enc=p_enc,
		                               tokenizer=tokenizer, batch_size=batch_size, sim_cache=sim_cache,
		                               resume_from_checkpoint=resume_from_checkpoint)
	elif args.sampler == ABSBM25:
		if accelerator.is_local_main_process:
			print("Use ABSBM25 Sampler")
		sampler = ABSBM25Sampler(dataset=train_set, tokenizer=tokenizer,
		                         resume_from_checkpoint=resume_from_checkpoint)
	elif args.sampler == Sequential:
		if accelerator.is_local_main_process:
			print("Use Sequential Sampler")
		sampler = BatchSampler(SequentialSampler(train_set), batch_size=batch_size, drop_last=True)
	else:
		if accelerator.is_local_main_process:
			print("Use Random Sampler")
		sampler = BatchSampler(RandomSampler(data_source=train_set), batch_size=batch_size, drop_last=True)

	train_dataloader = DataLoader(dataset=train_set, batch_sampler=sampler, num_workers=4, pin_memory=True,
	                              collate_fn=DistABSCollactor(tokenizer=tokenizer, padding='max_length'))
	eval_dataloader = DataLoader(dataset=dev_set, batch_size=eval_batch_size, num_workers=4, pin_memory=True,
	                             collate_fn=DistEvalCollactor(tokenizer=tokenizer, padding='max_length'))

	# Prepare DDP Training
	optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
		optimizer, train_dataloader, eval_dataloader
	)

	init_dataloader = None
	if args.mix:
		if args.sampler == ABSBM25:
			init_dataloader = train_dataloader
		else:
			init_sampler = ABSBM25Sampler(dataset=train_set, tokenizer=tokenizer)
			init_dataloader = DataLoader(dataset=train_set, batch_sampler=init_sampler, num_workers=4, pin_memory=True,
			                             collate_fn=DistABSCollactor(tokenizer=tokenizer, padding='max_length'))
		init_dataloader = accelerator.prepare(init_dataloader)


	# Resume Training State
	if resume_from_checkpoint:
		optimizer_state_dict = torch.load(os.path.join(resume_from_checkpoint, OPTIMIZER_STATE), map_location='cpu')
		optimizer.load_state_dict(optimizer_state_dict)
		state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))

	# Start Training
	if args.mix:
		state.max_steps = len(init_dataloader) + (args.epoch-1) * len(train_dataloader)
	else:
		state.max_steps = args.epoch * len(train_dataloader)
	train_progress_bar = tqdm(total=state.max_steps, disable=not accelerator.is_local_main_process)
	train_progress_bar.update(state.global_step)
	epoch_trained = state.global_step // len(train_dataloader)
	steps_to_skip = state.global_step % len(train_dataloader)
	skipped_step = 0
	for epoch in range(epoch_trained, args.epoch):
		dataloader = init_dataloader if args.mix and epoch == 0 else train_dataloader
		for step, batch in enumerate(dataloader):
			if skipped_step < steps_to_skip:
				skipped_step += 1
				continue

			model.train()
			optimizer.zero_grad()

			batch['passage']['input_ids'] = torch.flatten(batch['passage']['input_ids'], end_dim=1)
			batch['passage']['attention_mask'] = torch.flatten(batch['passage']['attention_mask'], end_dim=1)
			scores = model(**batch, scores_only=True)
			loss = F.cross_entropy(scores, batch['labels'])

			accelerator.backward(loss)
			optimizer.step()

			train_progress_bar.update()
			state.global_step += 1

			# Log train set metrics
			if state.global_step > 0 and state.global_step % logging_step == 0 and accelerator.is_local_main_process:
				metrics = compute_metrics.compute(scores, batch['labels'])
				metrics['loss'] = loss.item()
				metrics['epoch'] = state.global_step / state.max_steps * args.epoch
				tb_writer.log_metric(step=state.global_step, logs=metrics)

			# Evaluation
			if state.global_step > 0 and state.global_step % args.eval_step == 0:
				if accelerator.is_local_main_process:
					print("Start Evaluation")
				model.eval()
				all_loss, all_scores, all_labels = [], None, None
				for eval_batch in tqdm(eval_dataloader, disable=not accelerator.is_local_main_process):
					eval_batch['passage']['input_ids'] = torch.flatten(eval_batch['passage']['input_ids'], end_dim=1)
					eval_batch['passage']['attention_mask'] = torch.flatten(eval_batch['passage']['attention_mask'],
					                                                        end_dim=1)
					with torch.no_grad():
						scores = model(**eval_batch, scores_only=True)
					scores = accelerator.gather(scores)
					labels = accelerator.gather(eval_batch['labels'])

					all_loss.append(F.cross_entropy(scores, labels).item())
					all_scores = scores if all_scores is None else torch.cat([all_scores, scores], dim=0)
					all_labels = labels if all_labels is None else torch.cat([all_labels, labels], dim=0)

				metrics = compute_metrics.compute(all_scores, all_labels)
				metrics['loss'] = np.mean(all_loss)
				metrics['epoch'] = state.global_step / state.max_steps * args.epoch
				if accelerator.is_local_main_process:
					tb_writer.log_metric(step=state.global_step, logs=metrics, split='eval')
					tb_writer.log_parm(step=state.global_step, param={"dist": compute_metrics.Dist(all_scores)})

			# Save Checkpoint
			if (state.global_step > 0 and state.global_step % args.save_step == 0) or state.global_step == state.max_steps:
				accelerator.wait_for_everyone()
				if accelerator.is_local_main_process:
					checkpoint_dir = os.path.join(output_dir, f'checkpoint-{state.global_step}')
					print(f'save {checkpoint_dir}')
					os.makedirs(checkpoint_dir, exist_ok=True)
					unwrapped_model = accelerator.unwrap_model(model)
					accelerator.save(unwrapped_model.state_dict(), os.path.join(checkpoint_dir, WEIGHT_NAME))
					accelerator.save(optimizer.state_dict(), os.path.join(checkpoint_dir, OPTIMIZER_STATE))
					if args.sampler in [ABSEncode, ABSBeamSearch, ABSBM25] and not (args.mix and epoch == 0):
						sampler.save_checkpoint(checkpoint_dir)
					state.save_to_json(os.path.join(checkpoint_dir, TRAINER_STATE_NAME))
					clean_checkpoints(output_dir, save_total_limit)

		# Reset Sampler
		if args.sampler in [ABSEncode, ABSBeamSearch]:
			sim_cache = encode(accelerator, model, encode_dataloader)
			sampler.re_encode(cache=sim_cache)

		if args.sampler in [ABSEncode, ABSBeamSearch, ABSBM25]:
			sampler.reset()

		# Clean Up CUDA Memory
		init_dataloader = None


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Condenser training")
	parser.add_argument("--fp16", action="store_true", default=False, help="If passed, will use FP16 training.")
	parser.add_argument("--cpu", action="store_true", default=False, help="If passed, will train on the CPU.")
	parser.add_argument("--sampler", required=True, help="Sampler Type")
	parser.add_argument("--mix", action="store_true", default=False, help="If passed, will use mix training")
	parser.add_argument("--output", required=True, help="model output directory")
	parser.add_argument("--pretrain", required=True, help="pretrain model name")
	parser.add_argument("--train_size", required=True, type=int, help="train set size")
	parser.add_argument("--dev_size", required=True, type=int, help="dev set size")
	parser.add_argument("--eval_step", default=500, type=int, help="evaluation step")
	parser.add_argument("--save_step", default=500, type=int, help="save step")
	parser.add_argument("--epoch", required=True, type=int, help="number of epoch")


	args = parser.parse_args()
	config = {
		"lr": 5e-6,
		"batch_size": 8,
		"logging_step": 10,
		"eval_batch_size": 10,
		"save_total_limit": 3,
		"model_path": None,
		"resume_from_checkpoint": None,
		'output_dir': f"model_output/{args.output}",
	}
	print(args)
	print(config)
	training(config=config, args=args)

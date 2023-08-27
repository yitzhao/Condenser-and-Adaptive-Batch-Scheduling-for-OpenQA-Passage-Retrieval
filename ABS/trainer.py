from typing import Any, Dict, Union

import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import EvalPrediction
from transformers.file_utils import is_apex_available, is_datasets_available
from transformers.trainer import Trainer, is_sagemaker_mp_enabled
from transformers.trainer_pt_utils import nested_numpify, nested_detach, IterableDatasetShard

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward


class DenseTrainer(Trainer):
    """
    Trainer compatible to Adaptive Batch Sampler
    """
    def __init__(self, abs_sampler=None, abs_collator=None, **kwargs):
        super(DenseTrainer, self).__init__(**kwargs)
        self.abs_sampler = abs_sampler
        self.abs_collator = abs_collator

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self.args.train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        if self.abs_sampler:
            return DataLoader(
                train_dataset,
                batch_sampler=self.abs_sampler,
                collate_fn=self.abs_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return DataLoader(
                train_dataset,
                batch_size=self.args.train_batch_size,
                sampler=self._get_train_sampler(),
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)

        model.train()
        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            labels = nested_numpify(nested_detach(inputs['labels']))
            outputs = tuple(v for k, v in outputs.items() if k != 'loss')
            outputs = nested_numpify(nested_detach(outputs))
            if self.state.global_step % self.args.logging_steps == 0:
                metrics = self.compute_metrics(EvalPrediction(predictions=outputs, label_ids=labels))
                self.log(metrics)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


class DistributedTrainer(Trainer):
    def __init__(self, **kwargs):
        super(DistributedTrainer, self).__init__(**kwargs)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        inputs = self._prepare_inputs(inputs)

        model.train()
        if is_sagemaker_mp_enabled():
            scaler = self.scaler if self.do_grad_scaling else None
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps, scaler=scaler)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.autocast_smart_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            labels = nested_numpify(nested_detach(inputs['labels']))
            outputs = tuple(v for k, v in outputs.items() if k != 'loss')
            outputs = nested_numpify(nested_detach(outputs))
            if self.state.global_step % self.args.logging_steps == 0:
                metrics = self.compute_metrics(EvalPrediction(predictions=outputs, label_ids=labels))
                self.log(metrics)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()

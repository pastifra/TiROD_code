# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import os
import warnings
import numpy as np
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

from nanodet.data.batch_process import stack_batch_img
from nanodet.optim import build_optimizer
from nanodet.util import convert_avg_params, gather_results, mkdir

from ..model.arch import build_model
from ..model.weight_averager import build_weight_averager
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping

class PseudoLabelTrainingTask(LightningModule):
    """
    Important modules modified for the distillation training task.
    Training step

    """

    def __init__(self, cfg, evaluator=None):
        super(PseudoLabelTrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.teacher = build_model(cfg.model)
        #Freeze teacher network
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = "NanoDet"
        self.weight_averager = None
        
    def warp_boxes(self, M, boxes, width, height):
        n = len(boxes)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
                n * 4, 2
            )  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
            # clip boxes
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            return xy.astype(np.float32)
        else:
            return boxes



    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch

    def forward(self, x):
        x = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        batch = self._preprocess_batch_input(batch)
        preds = self.forward(batch["img"])
        results = self.model.head.post_process(preds, batch)
        return results

    def training_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        img = batch["img"]

        ### INFER NETWORK ###

        feat = self.teacher.backbone(img)
        fpn_feat = self.teacher.fpn(feat)
        head_out = self.teacher.head(fpn_feat)
        dets = self.teacher.head.post_process(head_out, batch)
        
        ### ADD PSEUDO LABELS TO TASK ###
        id = 0
        for img_id, img_dets in dets.items():
            boxes = []
            labels = []
            for label, bboxes in img_dets.items():
                for bbox in bboxes:
                    score = bbox[-1]
                    if score > 0.3:
                        x0, y0, x1, y1 = [float(i) for i in bbox[:4]]
                        
                        boxes.append([x0, y0, x1, y1])
                        labels.append(label)
            boxes = self.warp_boxes(batch['warp_matrix'][id], np.array(boxes), self.cfg.data.train.input_size[0], self.cfg.data.train.input_size[1])
            if(len(boxes) > 0):
                batch["gt_bboxes"][id] = np.concatenate((batch["gt_bboxes"][id],np.array(boxes)), axis=0)
                batch["gt_labels"][id] = np.concatenate((batch["gt_labels"][id],np.array(labels)), axis=0)
            id += 1

        ### STUDENT MODEL ###
        feat = self.model.backbone(img)
        fpn_feat = self.model.fpn(feat)
        head_out = self.model.head(fpn_feat)

        loss, loss_states = self.model.head.loss(head_out, batch)

        ### LOGGING ###
        if self.global_step % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                self.trainer.num_training_batches,
                memory,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )

            self.logger.info(log_msg)

        return loss

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))

    def validation_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        if self.weight_averager is not None:
            preds, loss, loss_states = self.avg_model.forward_train(batch)
        else:
            preds, loss, loss_states = self.model.forward_train(batch)

        if batch_idx % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                sum(self.trainer.num_val_batches),
                memory,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        dets = self.model.head.post_process(preds, batch)
        return dets

    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the
        outputs of all validation steps.Evaluating results
        and save best model.
        Args:
            validation_step_outputs: A list of val outputs

        """
        results = {}
        for res in validation_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            eval_results, table = self.evaluator.evaluate(
                all_results, self.cfg.save_dir, rank=self.local_rank
            )
            metric = eval_results[self.cfg.evaluator.save_key]
            # save best model
            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    os.path.join(best_save_path, "model_best.ckpt")
                )
                self.save_model_state(
                    os.path.join(best_save_path, "nanodet_model_best.pth")
                )
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                if self.local_rank < 1:
                    with open(txt_path, "a") as f:
                        f.write("Epoch:{}\n".format(self.current_epoch + 1))
                        for k, v in eval_results.items():
                            f.write("{}: {}\n".format(k, v))
                        f.write(table)

            else:
                warnings.warn(
                    "Warning! Save_key is not in eval results! Only save model last!"
                )
            self.logger.log_metrics(eval_results, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))

    def test_step(self, batch, batch_idx):
        dets = self.predict(batch, batch_idx)
        return dets

    def test_epoch_end(self, test_step_outputs):
        results = {}
        for res in test_step_outputs:
            results.update(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )
        if all_results:
            res_json = self.evaluator.results2json(all_results)
            json_path = os.path.join(self.cfg.save_dir, "results.json")
            json.dump(res_json, open(json_path, "w"))

            if self.cfg.test_mode == "val":
                eval_results, table = self.evaluator.evaluate(
                    all_results, self.cfg.save_dir, rank=self.local_rank
                )
                txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
                with open(txt_path, "a") as f:
                    for k, v in eval_results.items():
                        f.write("{}: {}\n".format(k, v))
                    f.write(table)
        else:
            self.logger.info("Skip test on rank {}".format(self.local_rank))

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        optimizer = build_optimizer(self.model, optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        scheduler = {
            "scheduler": build_scheduler(optimizer=optimizer, **schedule_cfg),
            "interval": "epoch",
            "frequency": 1,
        }
        return dict(optimizer=optimizer, lr_scheduler=scheduler)

    def optimizer_step(
        self,
        epoch=None,
        batch_idx=None,
        optimizer=None,
        optimizer_idx=None,
        optimizer_closure=None,
        on_tpu=None,
        using_lbfgs=None,
    ):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == "constant":
                k = self.cfg.schedule.warmup.ratio
            elif self.cfg.schedule.warmup.name == "linear":
                k = 1 - (
                    1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                ) * (1 - self.cfg.schedule.warmup.ratio)
            elif self.cfg.schedule.warmup.name == "exp":
                k = self.cfg.schedule.warmup.ratio ** (
                    1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                )
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * k

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        state_dict = (
            self.weight_averager.state_dict()
            if self.weight_averager
            else self.model.state_dict()
        )
        torch.save({"state_dict": state_dict}, path)

    # ------------Hooks-----------------
    def on_fit_start(self) -> None:
        if "weight_averager" in self.cfg.model:
            self.logger.info("Weight Averaging is enabled")
            if self.weight_averager and self.weight_averager.has_inited():
                self.weight_averager.to(self.weight_averager.device)
                return
            self.weight_averager = build_weight_averager(
                self.cfg.model.weight_averager, device=self.device
            )
            self.weight_averager.load_from(self.model)

    def on_train_epoch_start(self):
        self.model.set_epoch(self.current_epoch)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.weight_averager:
            self.weight_averager.update(self.model, self.global_step)

    def on_validation_epoch_start(self):
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)

    def on_test_epoch_start(self) -> None:
        if self.weight_averager:
            self.on_load_checkpoint({"state_dict": self.state_dict()})
            self.weight_averager.apply_to(self.model)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) != len(self.model.state_dict()):
                self.logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['model'] = self.model.state_dict()

import copy
import json
import os
import warnings
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

class OEWCTrainingTask(LightningModule):

    def __init__(self, cfg,  fisher, evaluator=None):
        super(OEWCTrainingTask, self).__init__()
        torch.cuda.empty_cache()
        self.cfg = cfg
        self.lambda_ewc = 100000
        self.model = build_model(cfg.model)
        self.evaluator = evaluator
        if self.evaluator is None:
            print("Warning! Evaluator is not provided!")
        self.save_flag = -10
        self.log_style = "NanoDet"
        self.weight_averager = None
        self._device = 'cuda:2'
        self.fisher_information_list = fisher
        self.optimal_params = {}

        self.model = self.model.to(self._device)

    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self._device) for img in batch_imgs]
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
    
    def compute_fisher_information(self, dataloader):
        self.model.eval()
        fisher_information = {}
        for name, param in self.model.named_parameters():
            fisher_information[name] = torch.zeros_like(param)
        print("Computing Fisher Information ... ")
        for batch in dataloader:
            batch = self._preprocess_batch_input(batch)
            img = batch["img"]
            self.model.zero_grad()
            output = self.model(img)
            loss = F.cross_entropy(output, output.max(1)[1])
            loss.backward()

            for name, param in self.model.named_parameters():
                fisher_information[name] += param.grad ** 2

        for name in fisher_information:
            fisher_information[name] /= len(dataloader)
            fisher_information[name] = fisher_information[name].to(self._device)
        print("... Done")
        self.fisher_information_list.append(fisher_information)

    def store_optimal_params(self):
        print("Storing Optimal parameters ...")
        self.optimal_params = {name: param.clone().to(self._device) for name, param in self.model.named_parameters()}
        print("... Done")

    
    def huber_loss(self, diff, delta=1.0):
        abs_diff = torch.abs(diff)
        quadratic = torch.min(abs_diff, torch.tensor(delta))
        linear = abs_diff - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        return loss
    
    def ewc_loss(self):
        loss = 0
        for task_idx in range(len(self.fisher_information_list)):
            fisher_information = self.fisher_information_list[task_idx]
            for name, param in self.model.named_parameters():
                if "backbone" in name or "fpn" in name or "head.cls_convs" in name:
                    if name in fisher_information:
                        fisher = fisher_information[name].to(self._device)
                        optimal_param = self.optimal_params[name].to(self._device)
                        loss += (fisher * (param - optimal_param) ** 2).sum()
        return self.lambda_ewc / 2 * loss
    
    def hueber_ewc_loss(self):
        loss = 0
        delta = 1.0  # You can adjust this value as needed
        for task_idx in range(len(self.fisher_information_list)):
            fisher_information = self.fisher_information_list[task_idx]
            optimal_params = self.optimal_params
            for name, param in self.model.named_parameters():
                if "backbone" in name or "fpn" in name or "head.cls_convs" in name:
                    if name in fisher_information:
                        fisher = fisher_information[name].to(self._device)
                        optimal_param = optimal_params[name].to(self._device)
                        diff = param - optimal_param
                        loss += (fisher * self.huber_loss(diff, delta)).sum()
        return self.lambda_ewc / 2 * loss
    
    def training_step(self, batch, batch_idx):
        batch = self._preprocess_batch_input(batch)
        img = batch["img"]

        feat = self.model.backbone(img)
        stud_fpn_feat = self.model.fpn(feat)

        # Extract Head Conv towers outputs
        stud_int_head_out = []
        for feat, cls_convs in zip(stud_fpn_feat, self.model.head.cls_convs):
            for conv in cls_convs:
                feat = conv(feat)
            stud_int_head_out.append(feat)

        # Extract Head cls outputs
        stud_head_out = self.model.head(stud_fpn_feat)
        stud_cls_preds, stud_reg_preds = stud_head_out.split(
            [self.cfg.model.arch.head.num_classes, 4 * (7 + 1)], dim=-1
        )

        ### NANODET LOSS ###
        loss, loss_states = self.model.head.loss(stud_head_out, batch)

        ### EWC LOSS ###
        ewc_loss = self.hueber_ewc_loss()
        total_loss = loss + ewc_loss

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
            
            self.scalar_summary("Train_loss/" + "ewc", "Train", ewc_loss, self.global_step)

            log_msg += "ewc_loss:{:.4f}| ".format(ewc_loss)
            self.logger.info(log_msg)

        self.scalar_summary("Train_loss/" + "total_loss", "Train", total_loss, self.global_step)

        return total_loss

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
        and save best models.
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
            # save best models
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
                    "Warning! Save_key is not in eval results! Only save models last!"
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
        self.logger.info("Saving models to {}".format(path))
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
                    "match the models"
                )
            else:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint['models'] = self.model.state_dict()

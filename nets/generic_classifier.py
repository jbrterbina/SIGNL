import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from pathlib import Path
import numpy as np

import utils.evaluation as eval
from nets.softmax_loss import SoftmaxLossFunction


class GenericClassifier(pl.LightningModule):
    def __init__(self, model, loss="softmax", cls_weight=True, sweep=False):
        super().__init__()
        self.model = model
        in_dim = 80
        self.criterion = SoftmaxLossFunction(in_dim=in_dim, num_classes=2)
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        return self.model(x)

    def _compute_loss_and_acc(self, batch, mode):
        x, _, _, y = batch
        y_hat, _ = self(x)
        loss, scores, _ = self.criterion(y_hat, y)

        return loss, y, scores, None, None

    def training_step(self, batch, batch_idx):
        loss, y, scores, _, preds = self._compute_loss_and_acc(batch, mode=True)
        self.train_step_outputs.append(
            {
                "key": batch[1],
                "y": y.cpu(),
                "scores": scores.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y, scores, _, preds = self._compute_loss_and_acc(batch, mode=True)
        self.validation_step_outputs.append(
            {
                "key": batch[1],
                "y": y.cpu(),
                "scores": scores.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, y, scores, _, preds = self._compute_loss_and_acc(batch, mode=False)
        self.test_step_outputs.append(
            {
                "key": batch[1],
                "y": y.cpu(),
                "scores": scores.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
        )
        return loss

    def _aggregate_step_outputs(self, outputs, stage="test"):
        y_hat = np.concatenate([x["scores"].float().numpy() for x in outputs])
        y = np.concatenate([x["y"] for x in outputs])

        file_name = np.concatenate([x["key"] for x in outputs])
        y_summary = ["bonafide" if label == 1 else "fake" for label in y]

        log, th = eval.compute_eer_ori(y, y_hat)
        return log["EER"], log["ACC"]

    def on_train_epoch_end(self):
        all_loss = [output["loss"] for output in self.train_step_outputs]
        avg_loss = np.mean(all_loss)
        self.log("train_loss", avg_loss)
        print("\033[92m####### TRAINING #######\033[0m")
        train_eer, train_acc = self._aggregate_step_outputs(
            self.train_step_outputs, "train"
        )
        self.train_step_outputs.clear()

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        all_loss = [output["loss"] for output in self.validation_step_outputs]
        avg_loss = np.mean(all_loss)
        self.log("valid_loss", avg_loss)
        print("\033[92m####### VALIDATION #######\033[0m")
        val_eer, val_acc = self._aggregate_step_outputs(
            self.validation_step_outputs, "val"
        )
        self.log("valid_eer", val_eer)
        self.validation_step_outputs.clear()

    def on_test_end(self):
        all_loss = [output["loss"] for output in self.test_step_outputs]
        avg_loss = np.mean(all_loss)
        print("\033[92m####### EVALUATION #######\033[0m")
        test_eer, _ = self._aggregate_step_outputs(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

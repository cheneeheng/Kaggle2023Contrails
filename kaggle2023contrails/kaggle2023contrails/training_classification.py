import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, AveragePrecision, PrecisionRecallCurve
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from neptune.types import File
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from .model.Convnextv2 import Convnextv2


class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Convnextv2(model_name=config["encoder_name"],
                                pretrained=True,
                                features_only=False)
        if self.config["loss"]["name"] == "BCELoss":
            self.loss_module = torch.nn.BCEWithLogitsLoss()
        else:
            raise ValueError("Unknown Loss... " + self.config["loss"]["name"])
        self.val_step_outputs = []
        self.val_step_labels = []
        self.ap = Accuracy(task="binary")
        self.acc = AveragePrecision(task="binary")
        self.prc = PrecisionRecallCurve(task="binary")

    def forward(self, batch):
        imgs = batch
        preds = self.model(imgs)
        return preds

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.config["optimizer_params"])

        if self.config["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                **self.config["scheduler"]["params"]["CosineAnnealingLR"],
            )
            lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        elif self.config["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                **self.config["scheduler"]["params"]["ReduceLROnPlateau"],
            )
            lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        elif self.config["scheduler"]["name"] == "cosine_with_hard_restarts_schedule_with_warmup":  # noqa
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                **self.config["scheduler"]["params"][self.config["scheduler"]["name"]],  # noqa
            )
            lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=16)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds)
        self.val_step_labels.append(labels)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_outputs)
        all_labels = torch.cat(self.val_step_labels)
        all_preds = torch.sigmoid(all_preds)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        ap = self.ap(all_preds, all_labels.long())
        acc = self.acc(all_preds, all_labels.long())
        self.log("val_ap", ap, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.prc(all_preds, all_labels.long())
        fig, ax = self.prc.plot()
        self.logger.experiment["val_precision_recall"].append(
            File.as_image(fig))
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)


def training(config, trn_ds, val_ds, ckpt_filename, neptune_logger):

    data_loader_train = DataLoader(
        trn_ds,
        batch_size=config["train_bs"],
        shuffle=True,
        num_workers=config["workers"],
    )
    data_loader_validation = DataLoader(
        val_ds,
        batch_size=config["valid_bs"],
        shuffle=False,
        num_workers=config["workers"],
    )

    model = LightningModule(config["model"])

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_acc",
        dirpath=config["output_dir"],
        mode="max",
        filename=ckpt_filename,
        save_top_k=1,
        verbose=1,
    )
    progress_bar_callback = TQDMProgressBar(
        refresh_rate=config["progress_bar_refresh_rate"]
    )
    early_stop_callback = EarlyStopping(**config["early_stop"])

    neptune_logger.log_model_summary(model, max_depth=-1)
    neptune_logger.log_hyperparams(params=config)

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback,
                   early_stop_callback, progress_bar_callback],
        # logger=[CSVLogger(save_dir=config["log_dir"] + f'/logs_f{fold}/'), neptune_logger],
        logger=neptune_logger,
        **config["trainer"],
    )
    trainer.fit(model, data_loader_train, data_loader_validation)

    key = neptune_logger._construct_path_with_prefix("model/best_model_path")
    neptune_logger.experiment[key].upload(checkpoint_callback.best_model_path)

    model_name = neptune_logger._get_full_model_name(
        checkpoint_callback.best_model_path, checkpoint_callback)
    key = f"{neptune_logger._construct_path_with_prefix('model/checkpoints')}/{model_name}"
    neptune_logger.experiment[key].upload(
        File(checkpoint_callback.best_model_path))

    del (
        data_loader_train,
        data_loader_validation,
        model,
        trainer,
        checkpoint_callback,
        progress_bar_callback,
        early_stop_callback,
    )

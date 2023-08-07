import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.functional import dice
import lightning.pytorch as pl
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from neptune.types import File
import segmentation_models_pytorch as smp
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from .model.UnetConvnextv2 import UnetConvnextv2
from .model.Mask2Former import Mask2Former

seg_models = {
    "Unet": smp.Unet,
    "Unet++": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3+": smp.DeepLabV3Plus,
    "UnetConvnextv2": UnetConvnextv2,
    "Mask2Former": Mask2Former
}


class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config["seg_model"] == "Mask2Former":
            _model = seg_models[config["seg_model"]](
                model_name=config["model_name"],
                num_labels=config.get("classes", 2),
                ignore_index=config.get("ignore_index", None),
                reduce_labels=config.get("reduce_labels", False),
            )
            self.collate_fn = _model.generate_collate_fn()
            self.model = _model.model
            self.processor = _model.processor
            self.post_process_semantic_segmentation = _model.post_process_semantic_segmentation  # noqa
        else:
            self.model = seg_models[config["seg_model"]](
                encoder_name=config["encoder_name"],
                encoder_weights="imagenet",
                decoder_channels=config["decoder_channels"],
                in_channels=3,
                classes=config.get("classes", 1),
                activation=None,
            )
            self.collate_fn = None
        if self.config["loss"]["name"] == "DiceLoss":
            self.loss_module = smp.losses.DiceLoss(
                mode="binary",
                smooth=config["loss"]["loss_smooth"]
            )
        elif self.config["loss"]["name"] == "FocalLoss":
            self.loss_module = smp.losses.FocalLoss(
                mode="binary",
                alpha=config["loss"]["alpha"],
                gamma=config["loss"]["gamma"],
                normalized=config["loss"]["normalized"]
            )
        else:
            raise ValueError("Unknown Loss... " + self.config["loss"]["name"])
        self.val_step_outputs = []
        self.val_step_labels = []

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

    def _maskformer_step(self, imgs, labels, postprocess=False):
        batch = self.processor(
            [i for i in imgs],
            segmentation_maps=[l for l in labels],
            return_tensors="pt",
        )
        outputs = self.model(
            pixel_values=batch["pixel_values"].to(self.model.device),
            mask_labels=[labels.to(self.model.device)
                         for labels in batch["mask_labels"]],
            class_labels=[labels.to(self.model.device)
                          for labels in batch["class_labels"]],
        )
        output_dict = {"outputs": outputs, "loss": outputs.loss}
        if postprocess:
            preds = self.post_process_semantic_segmentation(
                outputs, self.config["image_size"])
            preds = torch.stack(preds, axis=0)
            if self.config["image_size"] != 256:
                preds = F.interpolate(preds, size=256, mode='bilinear')
            output_dict["preds"] = preds
        return output_dict

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        if self.config["seg_model"] == "Mask2Former":
            output_dict = self._maskformer_step(imgs, labels)
            loss = output_dict["loss"]
        else:
            preds = self.model(imgs)
            if self.config["image_size"] != 256:
                preds = F.interpolate(preds, size=256, mode='bilinear')
            loss = self.loss_module(preds, labels)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, batch_size=16)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        if self.config["seg_model"] == "Mask2Former":
            output_dict = self._maskformer_step(imgs, labels)
            loss = output_dict["loss"]
            preds = output_dict["preds"]
        else:
            preds = self.model(imgs)
            if self.config["image_size"] != 256:
                preds = F.interpolate(preds, size=256, mode='bilinear')
            loss = self.loss_module(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_outputs.append(preds)
        self.val_step_labels.append(labels)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_outputs)
        all_labels = torch.cat(self.val_step_labels)
        if self.config["seg_model"] != "Mask2Former":
            all_preds = torch.sigmoid(all_preds)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        score = dice(all_preds, all_labels.long())
        self.log("val_dice", score, on_step=False,
                 on_epoch=True, prog_bar=True)
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)


def training(config, trn_ds, val_ds, ckpt_filename, neptune_logger):

    model = LightningModule(config["model"])

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

    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_dice",
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

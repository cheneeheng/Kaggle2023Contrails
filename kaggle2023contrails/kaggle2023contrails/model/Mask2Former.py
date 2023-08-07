import torch
from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation

# Based on:
# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/Fine-tuning/Fine_tuning_MaskFormerForInstanceSegmentation_on_semantic_sidewalk.ipynb

################################################################################
# Model arch
# 1. Swin transformer backbone => increase channel while decreasing spatial size
# 2. Swin decoder
#   => Projection layer
#   => swins
#   => split the final output (emb length channel) into 3 tensors (3 levels) + 1 fpn with lateral connection.
#   => linear projection for mask features from last fpn layer
# 3. Decoder
#   => use mask feature to predict mask.
#   => use transformer decoder with cross attention.
#       => query emb as Q, multilayer featuremaps as K and V.
#       => 1 pos emb for Q and mutliscale pos emb (sin) for K
#       => masked attention using the predicted mask
#       => output is used to repredict the mask.
#   => use the intermediate featuremaps (hidden features + LN) to predict class
#   => final mask and class logits are used.
################################################################################


class Mask2Former(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/mask2former-swin-tiny-ade-semantic",
        num_labels: int = 2,
        ignore_index: int | None = None,
        reduce_labels: bool = False,
    ):
        super().__init__()
        kwargs = {
            "do_resize": False,
            "do_normalize": False,
            "reduce_labels": reduce_labels,
        }
        if ignore_index is not None:
            kwargs["ignore_index"] = ignore_index
        self.processor = Mask2FormerImageProcessor.from_pretrained(
            model_name,
            num_labels=num_labels,
            **kwargs
        )
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        self.name = model_name.replace("/", "___")

    def generate_collate_fn(self) -> callable:
        def collate_fn(batch):
            inputs = list(zip(*batch))
            images = inputs[0]
            labels = inputs[1]
            batch = self.processor(
                images,
                segmentation_maps=labels,
                return_tensors="pt",
            )
            batch["original_images"] = inputs[0]
            batch["original_labels"] = inputs[1]
            return batch
        return collate_fn

    def forward(self, x):
        outputs = self.model(x)
        return outputs

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x


# model = Mask2Former()
# x = torch.rand(3, 256, 256)
# y = torch.eye(256, dtype=int)
# # y = torch.zeros(256, dtype=int)
# o = model.processor(x, segmentation_maps=y)
# # print(o["pixel_values"])

# with torch.no_grad():
#     outputs = model.model(
#         pixel_values=torch.tensor(o["pixel_values"]),
#         mask_labels=[labels for labels in o["mask_labels"]],
#         class_labels=[labels for labels in o["class_labels"]],
#     )

# # print(x)
# # print(model.model)
# print(y)
# print(o["mask_labels"])
# print(o["class_labels"])

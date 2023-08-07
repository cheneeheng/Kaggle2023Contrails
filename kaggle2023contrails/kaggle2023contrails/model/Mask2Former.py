import torch
from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation
from typing import Optional, List, Tuple

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

    @staticmethod
    def post_process_semantic_segmentation(
        outputs,
        original_sizes,
        target_sizes: Optional[List[Tuple[int, int]]] = None
    ) -> "torch.Tensor":
        """
        Converts the output of [`Mask2FormerForUniversalSegmentation`] into semantic segmentation maps. Only supports
        PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            target_sizes (`List[Tuple[int, int]]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
        Returns:
            `List[torch.Tensor]`:
                A list of length `batch_size`, where each item is a semantic segmentation map of shape (height, width)
                corresponding to the target_sizes entry (if `target_sizes` is specified). Each entry of each
                `torch.Tensor` correspond to a semantic class id.
        """
        class_queries_logits = outputs.class_queries_logits  # [batch_size, num_queries, num_classes+1]
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=original_sizes, mode="bilinear", align_corners=False
        )

        # Remove the null class `[..., :-1]`
        masks_classes = class_queries_logits.softmax(dim=-1)[..., :-1]
        # [batch_size, num_queries, height, width]
        masks_probs = masks_queries_logits.sigmoid()

        # Semantic segmentation logits of shape (batch_size, num_classes, height, width)
        segmentation = torch.einsum(
            "bqc, bqhw -> bchw", masks_classes, masks_probs)
        batch_size = class_queries_logits.shape[0]

        # Resize logits and compute semantic segmentation maps
        if target_sizes is not None:
            if batch_size != len(target_sizes):
                raise ValueError(
                    "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                )

            semantic_segmentation = []
            for idx in range(batch_size):
                resized_logits = torch.nn.functional.interpolate(
                    segmentation[idx].unsqueeze(dim=0), size=target_sizes[idx], mode="bilinear", align_corners=False
                )
                semantic_map = resized_logits[0].argmax(dim=0)
                semantic_segmentation.append(semantic_map)
        else:
            semantic_segmentation = segmentation.argmax(dim=1)
            semantic_segmentation = [semantic_segmentation[i]
                                     for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation


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

import timm
import torch
from PIL import Image
import requests
from transformers import AutoImageProcessor
from transformers import Mask2FormerForUniversalSegmentation
from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerConfig


model = Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-tiny-ade-semantic")
processor = Mask2FormerImageProcessor.from_pretrained(
    "facebook/mask2former-swin-tiny-ade-semantic")
print(processor)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")
# returns:
# pixel_values - model inputs (H, W, C)
# pixel_mask - 1 for valid pixels and 0 for invalid/padded pixels
# mask_labels
# class_labels

c = Mask2FormerConfig.from_pretrained(
    "facebook/mask2former-swin-tiny-ade-semantic")

with torch.no_grad():
    outputs = model(**inputs)

# # model predicts class_queries_logits of shape `(batch_size, num_queries)`
# # and masks_queries_logits of shape `(batch_size, num_queries, height, width)`
class_queries_logits = outputs.class_queries_logits
masks_queries_logits = outputs.masks_queries_logits

predicted_semantic_map = processor.post_process_semantic_segmentation(
    outputs, target_sizes=[image.size[::-1]])[0]

print("Finished...")

# print(list(outputs.__dict__.keys()))
# print(class_queries_logits.shape)
# print(masks_queries_logits.shape)
# print(predicted_semantic_map.shape)
# print(processor)
# print(model.model)
# print(list(model.__dict__.keys()))
# print(model.criterion)
# print(model.training)

# model = timm.create_model(
#     'vit_small_patch14_dinov2.lvd142m',
#     pretrained=True,
#     num_classes=0,  # remove classifier nn.Linear
# )
# # model = torch.nn.Sequential(*list(model.children())[:-1])
# model.norm = torch.nn.Identity()
# model.fc_norm = torch.nn.Identity()
# model.head_drop = torch.nn.Identity()
# model.head = torch.nn.Identity()
# print(model)

# # # get model specific transforms (normalization, resize)
# # data_config = timm.data.resolve_model_data_config(model)
# # transforms = timm.data.create_transform(**data_config, is_training=False)

# # # output is (batch_size, num_features) shaped tensor
# # output = model(transforms(img).unsqueeze(0))

# # # or equivalently (without needing to set num_classes=0)

# # output = model.forward_features(transforms(img).unsqueeze(0))
# # # output is unpooled, a (1, 1370, 384) shaped tensor

# # output = model.forward_head(output, pre_logits=True)
# # # output is a (1, num_features) shaped tensor

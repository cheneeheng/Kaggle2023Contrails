import torch
from PIL import Image
import timm
import pprint
import numpy as np

print(f"\ntimm.__version__ : {timm.__version__}")

timm_model_list = timm.list_models("convnextv2*", pretrained=True)
print(f"\ntimm_model_list :")
pprint.pp(timm_model_list)

img = Image.fromarray(np.ones((384, 384, 3), dtype=np.uint8))

convnextv2 = timm.create_model(
    "hf_hub:timm/convnextv2_tiny.fcmae_ft_in22k_in1k",
    #     "hf_hub:timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384",
    pretrained=True,
    features_only=True,
)
convnextv2 = convnextv2.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(convnextv2)
transforms = timm.data.create_transform(**data_config, is_training=False)
print(f"\ndata_config :")
pprint.pp(data_config)
print(f"\ntransforms :")
pprint.pp(transforms)

# unsqueeze single image into batch of 1
output = convnextv2(transforms(img).unsqueeze(0))
output = convnextv2(torch.tensor(np.ones((2, 3, 384, 384))).float())

print(f"\noutput shape :")
for o in output:
    pprint.pp(o.shape)

    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 40, 56, 56])
    #  torch.Size([1, 80, 28, 28])
    #  torch.Size([1, 160, 14, 14])
    #  torch.Size([1, 320, 7, 7])

    #  torch.Size([1, 96, 96, 96])
    #  torch.Size([1, 192, 48, 48])
    #  torch.Size([1, 384, 24, 24])
    #  torch.Size([1, 768, 12, 12])

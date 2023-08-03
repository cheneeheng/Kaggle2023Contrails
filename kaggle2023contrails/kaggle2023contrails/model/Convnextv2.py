import timm
import torch


class Convnextv2(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        pretrained=True,
        features_only=False,
        num_classes=1
    ):
        super().__init__()
        _convnext = timm.create_model(
            f"hf_hub:timm/{model_name}",
            pretrained=pretrained,
            features_only=features_only,
        )
        _convnext.head.fc = torch.nn.Linear(
            _convnext.head.fc.in_features,
            out_features=num_classes,
            bias=_convnext.head.fc.bias is not None,
            device=_convnext.head.fc.weight.device,
            dtype=_convnext.head.fc.weight.dtype
        )
        self.model = _convnext
        self.name = f"timm-{model_name}"

    def forward(self, x):
        return self.model(x)

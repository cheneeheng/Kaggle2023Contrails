import timm
import torch
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)
from segmentation_models_pytorch.decoders.unet.decoder import DecoderBlock
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
from segmentation_models_pytorch.decoders.unet.decoder import CenterBlock
from typing import Optional, Union, List


class UnetDecoderExt(UnetDecoder):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        use_batchnorm=True,
        attention_type=None,
        center=False,
    ):
        super().__init__(encoder_channels, decoder_channels,
                         n_blocks, use_batchnorm, attention_type, center)
        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )
        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0, 0]
        out_channels = decoder_channels
        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = torch.nn.Identity()
        kwargs = dict(use_batchnorm=use_batchnorm,
                      attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(self, *features):
        # reverse channels to start from head of encoder
        features = features[::-1]
        head = features[0]
        skips = features[1:]
        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


class UnetConvnextv2(SegmentationModel):
    # Baed on Unet
    def __init__(
        self,
        encoder_name: str = "convnextv2_tiny.fcmae_ft_in22k_in1k",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = None,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (192, 96, 48, 24, 12),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()
        _convnextv2 = timm.create_model(
            f"hf_hub:timm/{encoder_name}",
            pretrained=True,
            features_only=True,
        )
        _convnextv2 = _convnextv2.eval()
        self.encoder = _convnextv2
        self.encoder.output_stride = _convnextv2.feature_info.reduction()[-1]
        self.decoder = UnetDecoderExt(
            # (96, 192, 384, 768),
            encoder_channels=_convnextv2.feature_info.channels(),
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=decoder_attention_type,
        )
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
        self.name = f"u-timm-{encoder_name}"
        self.initialize()

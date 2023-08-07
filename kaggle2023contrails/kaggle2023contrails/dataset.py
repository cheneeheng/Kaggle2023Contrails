import torch
import numpy as np
import torchvision.transforms as T
from torchvision.transforms import functional as F


class RandomResizedCrop(T.RandomResizedCrop):
    # size,
    # scale=(0.08, 1.0),
    # ratio=(3.0 / 4.0, 4.0 / 3.0),
    # interpolation=InterpolationMode.BILINEAR,
    # antialias: Optional[Union[str, bool]] = "warn",
    def __init__(self, size, image_size, **kwargs):
        super().__init__(size, **kwargs)
        self.i = None
        self.j = None
        self.h = None
        self.w = None
        if isinstance(image_size, int):
            self.dummy_tensor = torch.zeros((3, image_size, image_size))
        else:
            self.dummy_tensor = torch.zeros((3, *image_size))

    def update_params(self):
        self.i, self.j, self.h, self.w = super().get_params(
            self.dummy_tensor, self.scale, self.ratio)

    def get_params(self):
        return self.i, self.j, self.h, self.w

    def set_params(self, *args):
        self.i, self.j, self.h, self.w = args

    def forward(self, img):
        return F.resized_crop(
            img, self.i, self.j, self.h, self.w,
            self.size, self.interpolation, antialias=self.antialias
        )


class ContrailsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 image_size=256,
                 mean: tuple | None = (0.485, 0.456, 0.406),
                 std: tuple | None = (0.229, 0.224, 0.225),
                 random_crop_resize: dict | None = None,
                 label_dtype: torch.dtype = torch.float,
                 classification: bool = False,
                 train: bool = True):
        self.df = df
        self.trn = train
        self.image_size = image_size
        self.classification = classification
        # Image -------
        self.image_transforms = T.Compose([
            T.ToTensor(),  # changes to [c,h,w]
            T.ConvertImageDtype(torch.float),
        ])
        if image_size != 256:
            self.image_transforms.transforms.append(T.Resize(image_size))
        if random_crop_resize is not None:
            self.image_transforms.transforms.append(
                RandomResizedCrop(size=image_size,
                                  image_size=image_size,
                                  **random_crop_resize)
            )
        self.image_transforms.transforms.append(T.Normalize(mean, std))
        # Label -------
        self.label_transforms = T.Compose([
            T.ToTensor(),  # changes to [1,h,w]
            T.ConvertImageDtype(torch.float),
        ])
        if random_crop_resize is not None:
            self.label_transforms.transforms.append(
                RandomResizedCrop(size=image_size,
                                  image_size=image_size,
                                  **random_crop_resize)
            )
        self.label_dtype = label_dtype
        # Misc -------
        self.rrc_image_idx = next(
            (i for i, x in enumerate(self.image_transforms.transforms)
             if isinstance(x, RandomResizedCrop)),
            None
        )
        self.rrc_label_idx = next(
            (i for i, x in enumerate(self.label_transforms.transforms)
             if isinstance(x, RandomResizedCrop)),
            None
        )

    def _update_random_resize_crop_params(self):
        if self.rrc_image_idx is not None:
            self.image_transforms.transforms[self.rrc_image_idx].update_params(
            )
            self.label_transforms.transforms[self.rrc_label_idx].set_params(
                *(self.image_transforms.transforms[self.rrc_image_idx].get_params()))

    def _get_classification_label(self, label):
        if self.classification:
            label = label.sum().clip(0, 1).unsqueeze(-1)
        return label

    def __getitem__(self, index):
        raw_data = np.load(str(self.df.iloc[index].path))
        image = raw_data[..., :-1]
        label = raw_data[..., -1]
        self._update_random_resize_crop_params()
        image = self.image_transforms(image)  # C,H,W
        label = self.label_transforms(label).squeeze(0)  # H,W
        label = self._get_classification_label(label)
        label = label.to(self.label_dtype)
        return image, label

    def __len__(self):
        return len(self.df)

import torch
import numpy as np
import torchvision.transforms as T


class ContrailsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 df,
                 image_size=256,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 train=True):
        self.df = df
        self.trn = train
        self.image_size = image_size
        self.image_transforms = T.Compose([
            T.ToTensor(),  # changes to [c,h,w]
            T.ConvertImageDtype(torch.float),
            T.Normalize(mean, std),
        ])
        if image_size != 256:
            self.image_transforms.transforms.insert(-1, T.Resize(image_size))
        self.label_transforms = T.Compose([
            T.ToTensor(),
            T.ConvertImageDtype(torch.float),
        ])

    def __getitem__(self, index):
        raw_data = np.load(str(self.df.iloc[index].path))
        image = raw_data[..., :-1]
        label = raw_data[..., -1]
        image = self.image_transforms(image)
        label = self.label_transforms(label).squeeze(0)
        return image, label

    def __len__(self):
        return len(self.df)

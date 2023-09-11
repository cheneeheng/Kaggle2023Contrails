# BASED ON : https://github.com/NielsRogge/Transformers-Tutorials/blob/master/MaskFormer/Fine-tuning/Fine_tuning_MaskFormerForInstanceSegmentation_on_semantic_sidewalk.ipynb

import numpy as np
import matplotlib.pyplot as plt
import json
import pprint
import albumentations as A
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import evaluate
from huggingface_hub import hf_hub_download
from datasets import load_dataset
from transformers import Trainer
from transformers import TrainingArguments
from transformers import Mask2FormerImageProcessor
from transformers import Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerForUniversalSegmentationOutput  # noqa


def color_palette():
    """Color palette that maps each class to RGB values.

    This one is actually taken from ADE20k.
    """
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]


SHOW_SAMPLE_DATA = False
SHOW_UNNORMALIZED_DATA = False
SHOW_UNNORMALIZED_BATCH_DATA = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def to_tensor(x): return torch.tensor(x, device=DEVICE)


TRAIN_MODE = "trainer"  # ["trainer", "pytorch"]

MODEL_REPO_NAME = "facebook/mask2former-swin-tiny-ade-semantic"
DATASET_REPO_NAME = "segments/sidewalk-semantic"
SEED = 1
TEST_SIZE = 0.01
LR = 5e-5
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255
PALETTE = color_palette()

filename = "id2label.json"
with open(hf_hub_download(DATASET_REPO_NAME, filename, repo_type="dataset"), "r") as f:
    id2label = json.load(f)
ID_TO_LABEL = {int(k): v for k, v in id2label.items()}
print("ID_TO_LABEL:")
pprint.pprint({k: (v1, v2)
               for ((k, v1), v2) in zip(ID_TO_LABEL.items(),
                                        [[-1, -1, -1]] + PALETTE)})

# 1. DATASET -------------------------------------------------------------------
dataset = load_dataset(DATASET_REPO_NAME)
print("Dataset overview:\n", dataset)
print(f"Original image size: {dataset['train'][0]['pixel_values'].size}")
print(f"Original image mode: {dataset['train'][0]['pixel_values'].mode}")

dataset = dataset.shuffle(seed=SEED)
dataset = dataset["train"].train_test_split(test_size=TEST_SIZE)
trn_ds = dataset["train"]
tst_ds = dataset["test"]

if SHOW_SAMPLE_DATA:
    example = trn_ds[1]
    image = np.array(example['pixel_values'])
    label = np.array(example['label'])
    label_rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for label_i, color in enumerate(PALETTE):
        label_rgb[label - 1 == label_i, :] = color
    img = image * 0.5 + label_rgb * 0.5
    img = img.astype(np.uint8)
    img = np.concatenate([image, img, label_rgb], axis=1)
    plt.figure(figsize=(18, 6))
    plt.imshow(img)
    plt.show()


class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, dataset, transform):
        """
        Args:
            dataset
        """
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        original_image = np.array(self.dataset[idx]['pixel_values'])
        original_label = np.array(self.dataset[idx]['label'])
        transformed = self.transform(image=original_image,
                                     mask=original_label)
        image, label = transformed['image'], transformed['mask']
        # convert to C, H, W
        image = image.transpose(2, 0, 1)
        return image, label, original_image, original_label


trn_transform = A.Compose([
    A.LongestMaxSize(max_size=1333),
    A.RandomCrop(width=64, height=64),
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),
])

tst_transform = A.Compose([
    A.Resize(width=64, height=64),
    A.Normalize(mean=ADE_MEAN, std=ADE_STD),

])

trn_pt_ds = ImageSegmentationDataset(trn_ds, transform=trn_transform)
tst_pt_ds = ImageSegmentationDataset(tst_ds, transform=tst_transform)

image, label, _, _ = trn_pt_ds[0]
print("Transformed image size:", image.shape)
print("Transformed label size:", label.shape)

if SHOW_UNNORMALIZED_DATA:
    unnorm_image = (image * np.array(ADE_STD)
                    [:, None, None]) + np.array(ADE_MEAN)[:, None, None]
    unnorm_image = (unnorm_image * 255).astype(np.uint8)
    unnorm_image = np.moveaxis(unnorm_image, 0, -1)
    label_rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for label_i, color in enumerate(PALETTE):
        label_rgb[label - 1 == label_i, :] = color
    img = unnorm_image * 0.5 + label_rgb * 0.5
    img = img.astype(np.uint8)
    img = np.concatenate([unnorm_image, img, label_rgb], axis=1)
    plt.figure(figsize=(18, 6))
    plt.imshow(img)
    plt.show()

preprocessor = Mask2FormerImageProcessor.from_pretrained(
    MODEL_REPO_NAME,
    num_labels=len(ID_TO_LABEL),
    ignore_index=0,
    reduce_labels=False,
    do_resize=False,
    do_rescale=False,
    do_normalize=False
)


def collate_fn(batch):
    inputs = list(zip(*batch))
    images = inputs[0]
    labels = inputs[1]
    # this function pads the inputs to the same size,
    # and creates a pixel mask
    # actually padding isn't required here since we are cropping
    batch = preprocessor(
        images,
        segmentation_maps=labels,
        return_tensors="pt",
    )
    batch["original_images"] = to_tensor(inputs[2])
    batch["original_labels"] = to_tensor(inputs[3])
    return batch


trn_dl = DataLoader(trn_pt_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)  # noqa
tst_dl = DataLoader(tst_pt_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)  # noqa

print("Sample batch:")
batch = next(iter(trn_dl))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    else:
        print(k, v[0].shape)

if SHOW_UNNORMALIZED_BATCH_DATA:
    image = batch["pixel_values"][0].numpy()
    label = np.sum(batch["mask_labels"][0].numpy() *
                   batch["class_labels"][0].numpy()[:, None, None], axis=0)
    unnorm_image = (image * np.array(ADE_STD)
                    [:, None, None]) + np.array(ADE_MEAN)[:, None, None]
    unnorm_image = (unnorm_image * 255).astype(np.uint8)
    unnorm_image = np.moveaxis(unnorm_image, 0, -1)
    label_rgb = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for label_i, color in enumerate(PALETTE):
        label_rgb[label - 1 == label_i, :] = color
    img = unnorm_image * 0.5 + label_rgb * 0.5
    img = img.astype(np.uint8)
    img = np.concatenate([unnorm_image, img, label_rgb], axis=1)
    plt.figure(figsize=(18, 6))
    plt.imshow(img)
    plt.show()


# 2. MODEL ---------------------------------------------------------------------

class LocalModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MODEL_REPO_NAME,
            num_labels=len(ID_TO_LABEL),
            # num_queries=num_queries,
            id2label=ID_TO_LABEL,
            ignore_mismatched_sizes=True
        )
        self.device = device

    def forward(self, **kwargs):
        if "mask_labels" in kwargs and "class_labels" in kwargs:
            outputs = self.model(
                pixel_values=kwargs["pixel_values"].to(self.device),
                mask_labels=[labels.to(self.device) for labels in kwargs["mask_labels"]],  # noqa
                class_labels=[labels.to(self.device) for labels in kwargs["class_labels"]],  # noqa
            )
            outputs.loss = outputs.loss.squeeze()
        else:
            outputs = self.model(
                pixel_values=kwargs["pixel_values"].to(self.device),
            )
        if not self.model.training:
            if "original_labels" in kwargs:
                labels = kwargs['original_labels']
                target_sizes = [label.shape for label in labels]
            elif "mask_labels" in kwargs and "class_labels" in kwargs:
                target_sizes = [ml[0].shape for ml in kwargs["mask_labels"]]
            else:
                target_sizes = None
            predicted_segmentation_maps = \
                preprocessor.post_process_semantic_segmentation(
                    outputs, target_sizes=target_sizes)
            outputs = {
                'loss': outputs.loss,
                'predicted_segmentation_maps': torch.stack(predicted_segmentation_maps, dim=0),  # noqa
                # 'class_queries_logits': outputs.class_queries_logits,
                # 'masks_queries_logits': outputs.masks_queries_logits,
                # 'encoder_last_hidden_state': outputs.encoder_last_hidden_state,
                # 'pixel_decoder_last_hidden_state': outputs.pixel_decoder_last_hidden_state,
                # 'transformer_decoder_last_hidden_state': outputs.transformer_decoder_last_hidden_state,
            }
        return outputs


model = LocalModel(DEVICE)

model_config = model.model.config
model_config.backbone_config.id2label = None
model_config.backbone_config.label2id = None
print("Model config:", model_config)

outputs = model(**batch)
print("Sample batch loss :", outputs['loss'])


# 3. TRAIN ---------------------------------------------------------------------
# NATIVE -----------------------------------------------------------------------
if TRAIN_MODE == "pytorch":

    model.to(DEVICE)

    metric = evaluate.load("mean_iou")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    running_loss = 0.0
    num_samples = 0
    for epoch in range(100):

        print("Epoch:", epoch)
        model.train()

        for idx, batch in enumerate(tqdm(trn_dl)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"].to(DEVICE),
                mask_labels=[labels.to(DEVICE) for labels in batch["mask_labels"]],  # noqa
                class_labels=[labels.to(DEVICE) for labels in batch["class_labels"]],  # noqa
            )

            # Backward propagation
            loss = outputs.loss
            loss.backward()

            batch_size = batch["pixel_values"].size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                print("Loss:", running_loss/num_samples)

            # Optimization
            optimizer.step()

        model.eval()
        for idx, batch in enumerate(tqdm(tst_dl)):
            if idx > 5:
                break

            # get ground truth segmentation maps
            if "original_labels" in batch:
                ground_truth_segmentation_maps = batch["original_labels"]
            elif "mask_labels" in batch and "class_labels" in batch:
                ground_truth_segmentation_maps = \
                    [torch.sum(_mask*_class[:, None, None], dim=0)
                     for _mask, _class in zip(batch["mask_labels"],
                                              batch["class_labels"])]
                ground_truth_segmentation_maps = \
                    torch.stack(ground_truth_segmentation_maps, dim=0)
            else:
                ground_truth_segmentation_maps = None

            # get input image
            pixel_values = batch["pixel_values"]

            # Forward pass
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values.to(DEVICE),
                                original_labels=ground_truth_segmentation_maps)

            # get predicted segmentation maps
            predicted_segmentation_maps = outputs['predicted_segmentation_maps']  # noqa

            metric.add_batch(references=ground_truth_segmentation_maps,
                             predictions=predicted_segmentation_maps)

        # NOTE this metric outputs a dict that also includes the mIoU per category
        # as keys so if you're interested, feel free to print them as well
        result = metric.compute(num_labels=len(id2label), ignore_index=0)
        print("Mean IoU:", result['mean_iou'])


# TRAINER ----------------------------------------------------------------------
elif TRAIN_MODE == "trainer":

    metric = evaluate.load("mean_iou")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predicted_segmentation_maps = logits
        # mask_labels = labels[0]
        # class_labels = labels[1]
        ground_truth_segmentation_maps = labels[2]
        return metric.compute(predictions=predicted_segmentation_maps,
                              references=ground_truth_segmentation_maps,
                              num_labels=len(id2label),
                              ignore_index=0)

    training_args = TrainingArguments(
        no_cuda=True,
        output_dir="test_trainer",
        report_to="none",
        logging_steps=1,
        # auto_find_batch_size=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=2,
        optim="adamw_torch",
        learning_rate=LR,
        evaluation_strategy="epoch",
        label_names=['mask_labels', 'class_labels', 'original_labels']
    )
    trainer = Trainer(
        model,
        training_args,
        train_dataset=trn_pt_ds,
        eval_dataset=tst_pt_ds,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # trainer.evaluate()

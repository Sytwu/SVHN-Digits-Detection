# dataset.py

import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import CocoDetection


class SVHNDataset(CocoDetection):
    def __init__(self, root, annFile, transforms=None):
        super().__init__(root, annFile, transforms=None)
        self.transforms = transforms

    def __getitem__(self, index):
        # Retrieve image, annotations, and image info
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        target_annots = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.root, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        # Calculate bounding box coordinates (top-left and bottom-right)
        # and collect labels
        boxes, labels = [], []
        for obj in target_annots:
            x_min, y_min, w, h = obj["bbox"]
            boxes.append([x_min, y_min, x_min + w, y_min + h])
            labels.append(obj["category_id"])

        # Default to original boxes and labels
        boxes_out, labels_out = boxes, labels

        # If transforms are provided, apply augmentation.
        # Otherwise, convert image to tensor directly
        if self.transforms:
            image_array = np.array(image)
            transformed = self.transforms(
                image=image_array,
                bboxes=boxes,
                labels=labels
            )

            # Use transformed results if bounding boxes remain
            if transformed["bboxes"]:
                image = transformed["image"]
                boxes_out = transformed["bboxes"]
                labels_out = transformed["labels"]
            else:
                image = transforms.ToTensor()(image)
        else:
            image = torchvision.transforms.functional.to_tensor(image)

        # Ensure consistent image dtype and
        # Convert target annotations to tensors
        if image.dtype != torch.float32:
            image = image.float() / 255.0

        target = {
            "boxes": torch.as_tensor(boxes_out, dtype=torch.float32),
            "labels": torch.as_tensor(labels_out, dtype=torch.int64)
        }

        return image, target


def Default_ShiftScaleRotate():
    ShiftScaleRotate = A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=15,
        p=0.5
    )
    return ShiftScaleRotate


def Default_ColorJitter():
    ColorJitter = A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.2,
        p=0.5
    )
    return ColorJitter


def Default_CoarseDropout():
    CoarseDropout = A.CoarseDropout(
        max_holes=8,
        max_height=16,
        max_width=16,
        min_holes=1,
        min_height=8,
        min_width=8,
        fill_value=0,
        p=0.5
    )
    return CoarseDropout


def Default_BboxParams():
    BboxParams = A.BboxParams(
        format='pascal_voc',
        label_fields=['labels'],
        clip=True,
        min_visibility=0.3
    )
    return BboxParams


def get_transform(usage=None):
    usage = usage.lower() if usage else ''

    train_veryhard_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        Default_ShiftScaleRotate(),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        Default_ColorJitter(),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        Default_CoarseDropout(),
        ToTensorV2()
    ], bbox_params=Default_BboxParams())

    train_hard_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        Default_ColorJitter(),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        ToTensorV2()
    ], bbox_params=Default_BboxParams())

    train_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        ToTensorV2()
    ], bbox_params=Default_BboxParams())

    train_simple_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ], bbox_params=Default_BboxParams())

    test_transforms = A.Compose([
        ToTensorV2()
    ], bbox_params=Default_BboxParams())

    postfix = 'Data Augmentation...'
    transforms_dict = {
        'train-veryhard': (train_veryhard_transforms, f"Very Hard {postfix}"),
        'train-hard': (train_hard_transforms, f"Hard {postfix}"),
        'train-simple': (train_simple_transforms, f"Simple {postfix}"),
        'train': (train_transforms, f"Normal {postfix}")
    }

    if 'train' in usage:
        for key in transforms_dict:
            if key != 'train' and key in usage:
                print(transforms_dict[key][1])
                return transforms_dict[key][0]

        print(transforms_dict['train'][1])
        return transforms_dict['train'][0]

    if 'valid' in usage or 'test' in usage:
        return test_transforms


def collate_fn(batch):
    return tuple(zip(*batch))

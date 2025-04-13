# main.py

import os
import torch

from models import (
    # FasterRCNN_ResNet50_FPN,
    FasterRCNN_ConvNeXt_FPN
)

from dataset import SVHNDataset, get_transform, collate_fn
from train import trainer
from utils import set_seed


os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

train_img_dir = "nycu-hw2-data/train"
train_ann_file = "nycu-hw2-data/train.json"
valid_img_dir = "nycu-hw2-data/valid"
valid_ann_file = "nycu-hw2-data/valid.json"


def get_dataset(usage, transform_usage):
    if usage == 'train':
        img_dir = train_img_dir
        ann_file = train_ann_file
    elif usage == 'valid':
        img_dir = valid_img_dir
        ann_file = valid_ann_file
    else:
        err_msg = f"Invalid usage '{usage}'. Expected 'train' or 'valid'."
        raise ValueError(err_msg)

    svhn_transform = get_transform(usage=transform_usage)
    svhn_dataset = SVHNDataset(img_dir, ann_file, svhn_transform)
    return svhn_dataset


def get_dataloader(dataset, batch_size=4, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return dataloader


def main():
    set_seed()

    train_dataset = get_dataset('train', 'train_veryhard')
    valid_dataset = get_dataset('valid', 'valid')

    train_loader = get_dataloader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = get_dataloader(valid_dataset, batch_size=4, shuffle=True)

    model = FasterRCNN_ConvNeXt_FPN(num_classes=11, pretrained=True).to(device)
    optimizer = model.get_optimizer(base_lr=3e-4, weight_decay=1e-5)
    lr_scheduler_fn = torch.optim.lr_scheduler.CosineAnnealingLR
    lr_scheduler = lr_scheduler_fn(optimizer, T_max=30)

    training_params = {
        'model': model,
        'train_loader': train_loader,
        'valid_loader': valid_loader,

        "optimizer": optimizer,
        "scheduler": lr_scheduler,

        'device': device,
        'num_epochs': 30,
        'save_path': 'svhn_fasterrcnn_convnext_BEST'
    }

    # Train the model
    results = trainer(training_params)

    # Access training results
    # trained_model = results["model"]
    best_mAP = results["best_mAP"]
    best_epoch = results["best_epoch"]
    history = results["history"]

    print('-'*50)
    print(f"Best mAP: {best_mAP:.4f} at Epoch {best_epoch}")
    print(history)


if __name__ == '__main__':
    main()

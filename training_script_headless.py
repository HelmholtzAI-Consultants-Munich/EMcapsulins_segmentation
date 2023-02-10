#@title
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import logging
import os
import shutil
import sys

import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar


import monai
from monai.transforms.compose import MapTransform
from monai.handlers import CheckpointSaver, MeanDice, StatsHandler, ValidationHandler, from_engine
#from monai.transforms import (
#    AsDiscreted,
#    CastToTyped,
#    LoadImaged,
#    Orientationd,
#    RandAffined,
#    RandFlipd,
#    RandGaussianNoised,
#    ScaleIntensityRanged,
#    Spacingd,
#    SpatialPadd,
#    )

from monai.transforms import (LoadImaged, EnsureChannelFirstd, EnsureTyped, Lambdad, RandGaussianNoised, RandFlipd,
                              RandAffined, SpatialPadd, ScaleIntensityRangePercentilesd, ToTensord,RandSpatialCropSamplesd, AsDiscreted)

from monai.networks.nets import BasicUNet




crop_size = (512, 512)



def get_xforms(mode="train", keys=("image", "label")):
    """returns a composed transform for train/val/infer."""

    xforms = [
        # << PREPROCESSING transforms >>
        LoadImaged(keys=keys),
        EnsureChannelFirstd(keys=keys),
        EnsureTyped(keys=keys),
        Lambdad(keys, np.nan_to_num),
        RandGaussianNoised(keys=[keys[0]], prob=0.2, std=0.01),
        RandFlipd(keys=keys, spatial_axis=0, prob=0.5),
        RandFlipd(keys=keys, spatial_axis=1, prob=0.5),
        RandAffined(
            keys=keys,
            prob=0.25,
            rotate_range=(0.5, 0.5, None),
            scale_range=(0.5, 0.5, None),
            mode=("bilinear", "nearest"),
        ),

      # compute patch crops
      SpatialPadd(  # ensure dimensions
                  keys=keys, spatial_size=crop_size, mode="reflect"),
      RandSpatialCropSamplesd(
          keys=keys,
          roi_size=crop_size,
          num_samples=20,
          random_size=False,
          random_center=True,
          ),
    
      ScaleIntensityRangePercentilesd(
              keys=[keys[0]],
              lower=0.5,
              upper=99.5,
              b_min=0,
              b_max=1,
              clip=True,
              relative=False,
              channel_wise=False,
          ),
          # make tensor
      ToTensord(keys=keys),
      ]
    return monai.transforms.Compose(xforms)


def get_net():
    """returns a unet model instance."""
    model = BasicUNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=8,
    features=(32, 32, 64, 128, 256, 32),
    dropout=0.1,
    act="mish",
    )
    return model


def get_inferer(_mode=None):
    """returns a sliding window inference instance."""

    patch_size = (512, 512)
    sw_batch_size, overlap = 2, 0.5
    inferer = monai.inferers.SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sw_batch_size,
        overlap=overlap,
        mode="gaussian",
        padding_mode="replicate",
    )
    return inferer


class DiceCELoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss(to_onehot_y=True, softmax=True)
        self.cross_entropy = nn.CrossEntropyLoss()

#    def forward(self, y_pred, y_true):
#        #import pdb;pdb.set_trace()
#        dice = self.dice(y_pred, y_true)
#         CrossEntropyLoss target needs to have shape (B, D, H, W)
#        # Target from pipeline has shape (B, 1, D, H, W)
#        cross_entropy = self.cross_entropy(y_pred, torch.squeeze(y_true, dim=1).long())
#        return dice + cross_entropy
    
    def forward(self, y_pred, y_true):
        y_true = y_true.argmax(dim=1, keepdim=True)
        dice = self.dice(y_pred, y_true)
        cross_entropy = self.cross_entropy(y_pred, y_true.argmax(dim=1))
        return dice + cross_entropy

import pdb
    
def train(data_folder=".", model_folder="runs"):
    """run a training pipeline."""
    training_images = sorted(glob.glob(os.path.join(data_folder, "**/*_mic.tif"),recursive=True))
    training_labels = sorted(glob.glob(os.path.join(data_folder, "**/*_label.tif"),recursive=True))
    assert len(training_images)==len(training_labels), "Number of training images != training labels"
    validation_images = training_images #TODO: Explain this 
    validation_labels = training_labels
    logging.info(f"training: image/label ({len(training_images)}) folder: {data_folder}")

    amp = True  # auto. mixed precision
    keys = ("image", "label")
    #train_frac, val_frac = 0.8, 0.2
    #n_train = int(train_frac * len(images)) + 1
    #n_val = min(len(images) - n_train, int(val_frac * len(images)))
    #logging.info(f"training: train {n_train} val {n_val}, folder: {data_folder}")

    train_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(training_images, training_labels)]
    val_files = [{keys[0]: img, keys[1]: seg} for img, seg in zip(validation_images, validation_labels)]

    # create a training data loader
    batch_size = 1
    logging.info(f"batch size {batch_size}")
    train_transforms = get_xforms("train", keys)
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms)
    train_loader = monai.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # create a validation data loader
    val_transforms = get_xforms("val", keys)
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms)
    val_loader = monai.data.DataLoader(
        val_ds,
        batch_size=1,  # image-level batch to the sliding window method, not the window-level batch
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # create BasicUNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = get_net().to(device)
    max_epochs, lr, momentum = 500, 1e-4, 0.95
    logging.info(f"epochs {max_epochs}, lr {lr}, momentum {momentum}")
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    # create evaluator (to be used to measure model quality during training
    #val_post_transform = monai.transforms.Compose(
    #    [AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=2)]
    #)
    val_post_transform = monai.transforms.Compose(
        [AsDiscreted(keys=("pred", "label"), argmax=(True, True), to_onehot=8)]
    )
    #val_post_transform = monai.transforms.Compose(
    #[
    #    lambda x:pdb.set_trace,
    #    lambda x: {"pred": x[0], "label": x[1].argmax(dim=1, keepdim=True)},
    #    AsDiscreted(keys=("pred", "label"), argmax=(True, False), to_onehot=2),
    #    
    #]
    #)
    
    
    val_handlers = [
        ProgressBar(),
        CheckpointSaver(save_dir=model_folder, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=3),
    ]
    evaluator = monai.engines.SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=get_inferer(),
        postprocessing=val_post_transform,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=from_engine(["pred", "label"]))
        },
        val_handlers=val_handlers,
        amp=amp,
    )

    # evaluator as an event handler of the trainer
    train_handlers = [
        ValidationHandler(validator=evaluator, interval=1, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=from_engine(["loss"], first=True)),
    ]
    trainer = monai.engines.SupervisedTrainer(
        device=device,
        max_epochs=max_epochs,
        train_data_loader=train_loader,
        network=net,
        optimizer=opt,
        loss_function=DiceCELoss(),
        inferer=get_inferer(),
        key_train_metric=None,
        train_handlers=train_handlers,
        amp=amp,
    )
    trainer.run()



if __name__ == "__main__":
    """
    Usage:
        python training_script_headless.py --data_folder "PATH/TO/FOLDER/Train" --model_folder  "PATH/TO/FOLDER/model"# run the training pipeline
       
    """
    parser = argparse.ArgumentParser(description="Run a basic UNet segmentation baseline.")
    parser.add_argument("--data_folder", default="", type=str, help="training data folder")
    parser.add_argument("--model_folder", default="model", type=str, help="model folder")
    args = parser.parse_args()

    monai.config.print_config()
    monai.utils.set_determinism(seed=0)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    data_folder = args.data_folder or os.path.join(os.getcwd(), "Train")
    train(data_folder=data_folder, model_folder=args.model_folder)



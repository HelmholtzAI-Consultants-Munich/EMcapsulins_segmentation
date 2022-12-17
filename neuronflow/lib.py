# basics
import os
import numpy as np
from path import Path
from tqdm import tqdm
import time

# dl
import torch
from torch.utils.data import DataLoader

import monai
from monai.networks.nets import BasicUNet
from monai.data import list_data_collate
from monai.inferers import SlidingWindowInferer

from monai.transforms import RandGaussianNoised
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambdad,
    ScaleIntensityRangePercentilesd,
    ToTensord,
)

# custom
from neuronflow.output import create_output_files
from neuronflow.postprocessing import postprocess


# GO
def inference_to_directory(
    microscopy_file,
    # segmentation_file,
    output_folder,
    prefix,
    binary_threshold=0.5,
    cuda_devices="1",
    tta=True,
    sliding_window_batch_size=32,
    sliding_window_overlap=0.5,
    workers=0,
    crop_size=(512, 512),
    model_weights="model_weights/heavy_weights.tar",
    verbosity=True,
):
    """
    call this function to run the sliding window inference.

    Parameters:
    niftis: list of nifti files to infer
    comment: string to comment
    model_weights: Path to the model weights
    tta: whether to run test time augmentations
    threshold: threshold for binarization of the network outputs. Greater than <theshold> equals foreground
    cuda_devices: which cuda devices should be used for the inference.
    crop_size: crop size for the inference
    workers: how many workers should the data loader use
    sw_batch_size: batch size for the sliding window inference
    overlap: overlap used in the sliding window inference

    see the above function definition for meaningful defaults.
    """
    # ~~<< S E T T I N G S >>~~
    torch.multiprocessing.set_sharing_strategy("file_system")

    # device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
    multi_gpu = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # clean memory
    torch.cuda.empty_cache()

    # T R A N S F O R M S
    inference_transforms = Compose(
        [
            LoadImageD(keys=["images"]),
            EnsureChannelFirstd(keys=["images"]),
            EnsureTyped(keys=["images"]),
            Lambdad(["images"], np.nan_to_num),
            ScaleIntensityRangePercentilesd(
                keys=["images"],
                lower=0.5,
                upper=99.5,
                b_min=0,
                b_max=1,
                clip=True,
                relative=False,
                channel_wise=True,
            ),
            ToTensord(keys=["images"]),
        ]
    )
    # D A T A L O A D E R
    dicts = []

    images = [microscopy_file]

    the_dict = {
        "image_path": microscopy_file,
        "images": images,
    }

    dicts.append(the_dict)

    # datasets
    inf_ds = monai.data.Dataset(data=dicts, transform=inference_transforms)

    # dataloaders
    data_loader = DataLoader(
        inf_ds,
        batch_size=1,
        num_workers=workers,
        collate_fn=list_data_collate,
        shuffle=False,
    )

    # ~~<< M O D E L >>~~
    model = BasicUNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=8,
        features=(32, 32, 64, 128, 256, 32),
        dropout=0.1,
        act="mish",
    )

    model_weights = Path(os.path.abspath(model_weights))
    checkpoint = torch.load(model_weights, map_location="cpu")

    # inferer
    patch_size = crop_size

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=sliding_window_batch_size,
        sw_device=device,
        device=device,
        overlap=sliding_window_overlap,
        mode="gaussian",
        padding_mode="replicate",
    )

    # send model to device // very important for optimizer to work on CUDA
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # load
    model.load_state_dict(checkpoint["model_state"])

    # epoch stuff
    if verbosity == True:
        time_date = time.strftime("%Y-%m-%d_%H-%M-%S")
        print("start:", time_date)

    # limit batch length?!
    batchLength = 0

    # eval!
    with torch.no_grad():
        model.eval()
        # loop through batches
        for counter, data in enumerate(tqdm(data_loader, 0)):
            if batchLength != 0:
                if counter == batchLength:
                    break

            # get the inputs and labels
            inputs = data["images"].float().to(device)

            outputs = inferer(inputs, model)
            # test time augmentations
            # TODO rethink dimensions
            if tta == True:
                n = 1.0
                for _ in range(4):
                    # test time augmentations
                    _img = RandGaussianNoised(keys="images", prob=1.0, std=0.001)(data)[
                        "images"
                    ]

                    output = inferer(_img.to(device), model)
                    outputs = outputs + output
                    n = n + 1.0
                    for dims in [[2], [3]]:
                        # for dims in [[3]]:
                        flip_pred = inferer(
                            torch.flip(_img.to(device), dims=dims), model
                        )
                        output = torch.flip(flip_pred, dims=dims)
                        outputs = outputs + output
                        n = n + 1.0
                outputs = outputs / n

            if verbosity == True:
                print("inputs shape:", inputs.shape)
                print("outputs:", outputs.shape)
                print("data length:", len(data))
                print("outputs shape 0:", outputs.shape[0])

            # loop through elements in batch
            for element in range(outputs.shape[0]):

                print("** processing:", data["image_path"][element])

                # output_folder = (
                #     testing_session_path
                #     + "/segmentations/"
                #     + str(data["exam"][element])
                # )
                os.makedirs(output_folder, exist_ok=True)

                onehot_model_output = outputs[element]

                # identity = output_folder + "/" + data["exam"][element]
                identity = output_folder + "/" + prefix

                segmentation_file = identity + "_segmentation.nii.gz"

                create_output_files(
                    onehot_model_outputs_CHW=onehot_model_output,
                    output_file=segmentation_file,
                    binary_output_file=identity + "_binary-segmentation.nii.gz",
                    binary_threshold=binary_threshold,
                    background_output_file=identity + "_out-bg.nii.gz",
                    foreground_output_file=identity + "_out-fg.nii.gz",
                    mQt_output_file=identity + "_out-1M-Qt.nii.gz",
                    mmQt_output_file=identity + "_out-2M-Qt.nii.gz",
                    mmmQt_output_file=identity + "_out-3M-Qt.nii.gz",
                    mMx_output_file=identity + "_out-1M-Mx.nii.gz",
                    mmMx_output_file=identity + "_out-2M-Mx.nii.gz",
                    mTm_output_file=identity + "_out-1M-Tm.nii.gz",
                    mQtTm_output_file=identity + "_out-1M-Qt-Tm.nii.gz",
                )

                postprocess(
                    raw_segmentation_file=segmentation_file,
                    polished_segmentation_file=identity + "_postprocessed.nii.gz",
                )

    if verbosity == True:
        print("end:", time.strftime("%Y-%m-%d_%H-%M-%S"))


if __name__ == "__main__":
    pass

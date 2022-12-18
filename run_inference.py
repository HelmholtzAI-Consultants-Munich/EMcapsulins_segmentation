from tqdm import tqdm

from neuronflow.utils import turbopath
from neuronflow.lib import single_inference
from neuronflow.postprocessing import postprocess


input_folder = turbopath("example_data/example_inputs")
target_folder = turbopath("example_data/example_outputs")

microscopy_images = input_folder.files()  # find all images

# loop through images
for img in tqdm(microscopy_images):
    identity = img.name[:-4]  # image name to use as prefix

    segmentation_file = identity + "_segmentation.nii.gz"

    single_inference(
        microscopy_file=img,
        segmentation_file=segmentation_file,
        # everything from here is optional and can be adjusted to your needs
        binary_segmentation_file=identity + "_binary-segmentation.nii.gz",
        binary_threshold=0.5,
        background_output_file=identity + "_out-bg.nii.gz",
        foreground_output_file=identity + "_out-fg.nii.gz",
        mQt_output_file=identity + "_out-1M-Qt.nii.gz",
        mmQt_output_file=identity + "_out-2M-Qt.nii.gz",
        mmmQt_output_file=identity + "_out-3M-Qt.nii.gz",
        mMx_output_file=identity + "_out-1M-Mx.nii.gz",
        mmMx_output_file=identity + "_out-2M-Mx.nii.gz",
        mTm_output_file=identity + "_out-1M-Tm.nii.gz",
        mQtTm_output_file=identity + "_out-1M-Qt-Tm.nii.gz",
        cuda_devices="1",
        tta=True,
        sliding_window_batch_size=32,
        sliding_window_overlap=0.5,
        workers=0,
        crop_size=(512, 512),
        model_weights="model_weights/heavy_weights.tar",
        verbosity=True,
    )

    # optional postprocessing to reproduce the manuscript's results
    postprocess(
        raw_segmentation_file=segmentation_file,
        polished_segmentation_file=identity + "_postprocessed.nii.gz",
        # everything from here is optional and can be adjusted to your needs
        prune_threshold=42,
        majority_vote=True,
        max_mav_threshold=500,
        circularity_threshold=0.2,
        fill_holes=True,
        debug=False,
    )

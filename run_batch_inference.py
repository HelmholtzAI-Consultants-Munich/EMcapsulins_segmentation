from tqdm import tqdm
import os,path

from neuronflow.utils import turbopath
from neuronflow.lib import single_inference
from neuronflow.postprocessing import postprocess

from tqdm import tqdm

from neuronflow.utils import turbopath
from neuronflow.lib import single_inference
from neuronflow.postprocessing import postprocess

input_folder = turbopath("example_data/example_inputs")
target_folder = turbopath("example_data/example_outputs")

import PIL
from PIL import Image
import numpy as np
from PIL import ImageOps


def run_batch_inference(image_folder, output_path, model_path="model_weights/heavy_weights.tar"):
    for filename in os.listdir(image_folder):
        if not filename.endswith("_mic.tif"):
            continue

        image_path = os.path.join(image_folder, filename)
        identity = filename.rstrip("_mic.tif").split("/")[-1]
        out_identity = os.path.join(path.Path(output_path),identity, identity)
        single_out_path = os.path.join(path.Path(output_path),identity)
        if not os.path.exists(single_out_path):
            os.makedirs(single_out_path)

        img = PIL.Image.open(image_path).convert("L") # L means grayscale
        img_data = np.array(img)
        img_data = img_data.astype('uint8')
        img.save(out_identity + "_mic.tif")        

        segmentation_file = out_identity + "_label.tif"

        single_inference(
            microscopy_file=out_identity + "_mic.tif",
            segmentation_file=segmentation_file,
            # everything from here is optional and can be adjusted to your needs
            binary_segmentation_file=out_identity + "_binary-segmentation.tif",
            binary_threshold=0.5,
            background_output_file=out_identity + "_out-bg.tif",
            foreground_output_file=out_identity + "_out-fg.tif",
            mQt_output_file=out_identity + "_out-1M-Qt.tif",
            mmQt_output_file=out_identity + "_out-2M-Qt.tif",
            mmmQt_output_file=out_identity + "_out-3M-Qt.tif",
            mMx_output_file=out_identity + "_out-1M-Mx.tif",
            mmMx_output_file=out_identity + "_out-2M-Mx.tif",
            mTm_output_file=out_identity + "_out-1M-Tm.tif",
            mQtTm_output_file=out_identity + "_out-1M-Qt-Tm.tif",
            cuda_devices="0",
            tta=True,
            sliding_window_batch_size=2,
            sliding_window_overlap=0.5,
            workers=0,
            crop_size=(512, 512),
            model_weights=model_path,
            verbosity=True,
        )
	
        postprocess(
            raw_segmentation_file=segmentation_file,
            polished_segmentation_file=out_identity + "_postprocessed.tif",
            # everything from here is optional and can be adjusted to your needs
            prune_threshold=42,
            majority_vote=True,
            mav_max_threshold=500,
            mav_circularity_threshold=0.2,
            fill_holes = True,
            )
            
run_batch_inference(input_folder,target_folder)          



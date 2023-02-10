import cc3d
import numpy as np
from neuronflow.utils import turbopath, read_tif, write_tif
from statistics import multimode

from skimage import measure

from monai.transforms import Compose
from monai.transforms.post.array import FillHoles

import tifffile as tiff


def get_circularity(numpy_array):
    props = measure.regionprops(numpy_array, numpy_array)

    area = props[0]["area"]
    perimeter = props[0]["perimeter"]

    if perimeter != 0:
        circularity = _calculate_circularity(perimeter, area)
    else:
        circularity = 1

    return circularity


def _calculate_circularity(perimeter, area):
    """Calculate the circularity of the region

    # From https://github.com/napari/napari/blob/5cfcc38c0a313f42cc8b0f82ac3db945874ae362/examples/annotate_segmentation_with_text.py#L75

    Parameters
    ----------
    perimeter : float
        the perimeter of the region
    area : float
        the area of the region
    Returns
    -------
    circularity : float
        The circularity of the region as defined by 4*pi*area / perimeter^2
    """
    if perimeter != 0:
        circularity = 4 * np.pi * area / (perimeter**2)
    else:
        circularity = 0

    return circularity


def postprocess(
    raw_segmentation_file,
    polished_segmentation_file,
    prune_threshold=42,
    majority_vote=True,
    mav_max_threshold=500,
    mav_circularity_threshold=0.2,
    fill_holes=True,
):
    """
    Postprocessing for emcapsulin segmentation with multiple steps:

    * compute connected component anlysis on foreground segmentation
    * prune components that are too small: -> n < 60 pixels
    * loop through remaining components and perform conditional majority vote.
    * filling holes in the segmentation maps
    """
    raw_segmentation_file = turbopath(raw_segmentation_file)
    polished_segmentation_file = turbopath(polished_segmentation_file)


    raw_segmentation = tiff.imread(raw_segmentation_file)
    raw_segmentation = raw_segmentation.astype(dtype=np.int32)

    binary_segmentation = raw_segmentation > 0

    ### 1. PRUNE ###
    if prune_threshold > 0:
        pruned_map = cc3d.dust(
            binary_segmentation,
            threshold=prune_threshold,
            connectivity=26,
            in_place=False,
        )

        multi_segmentation = raw_segmentation * pruned_map
        binary_segmentation = binary_segmentation * pruned_map
    else:
        multi_segmentation = raw_segmentation

    ### ** Convert to integer ** ###
    multi_segmentation = multi_segmentation.astype(dtype=np.int32)

    ### 2. Connected component analysis ###
    components, N = cc3d.connected_components(binary_segmentation, return_N=True)
    print(N)

    ### 3. Majority Voting! ###
    if majority_vote == True:
        for segid in range(1, N + 1):
            instance_mask = components == segid
            extracted_image = multi_segmentation * instance_mask
            # majority vote only if the object is not too big
            if np.count_nonzero(extracted_image) < mav_max_threshold:

                # check if circular enough
                circularity = get_circularity(
                    numpy_array=extracted_image.astype(dtype=np.int32)
                )
                if circularity > mav_circularity_threshold:

                    # majority vote only if unique most frequent value occurs
                    most_frequent_value = multimode(
                        extracted_image[extracted_image > 0]
                    )
                    if len(most_frequent_value) == 1:
                        multi_segmentation[extracted_image > 0] = most_frequent_value

    ### 4. Fill holes ###
    if fill_holes == True:
        print(tuple(np.unique(multi_segmentation)))

        hole_fill_tfs = Compose(
            [FillHoles(applied_labels=tuple(np.unique(multi_segmentation)))]
        )

        # add channel dimension
        multi_segmentation = np.expand_dims(multi_segmentation, axis=0)
        multi_segmentation = np.array(hole_fill_tfs(multi_segmentation))
        multi_segmentation = multi_segmentation[0]  # get rid of channel dimension again

    ### ** Save tif ** ###
    write_tif(
        numpy_array=multi_segmentation,
        output_tif_path=polished_segmentation_file,
    )
    return True

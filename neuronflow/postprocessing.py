import cc3d
import numpy as np
from neuronflow.utils import turbopath, read_nifti, write_nifti
from statistics import multimode

from skimage import measure

from monai.transforms import Compose
from monai.transforms.post.array import FillHoles


def get_circularity(numpy_array, debug):
    props = measure.regionprops(numpy_array, numpy_array)

    area = props[0]["area"]
    perimeter = props[0]["perimeter"]

    if perimeter != 0:
        circularity = _calculate_circularity(perimeter, area)
    else:
        circularity = 1

    if debug == True:
        print("perimeter=0")
        print("circularity:", circularity)

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


def _debug_cc3d(numpy_array, debug, stats=False):
    if debug == True:
        labels_out, N = cc3d.connected_components(numpy_array, return_N=True)
        uniques = np.unique(numpy_array)
        print(uniques)
        print(N)

        if stats == True:
            stats = cc3d.statistics(labels_out)
            print(stats)


def postprocess(
    raw_segmentation_file,
    polished_segmentation_file,
    prune_threshold=42,
    majority_vote=True,
    max_mav_threshold=500,
    circularity_threshold=0.2,
    fill_holes=True,
    debug=False,
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

    raw_segmentation = read_nifti(raw_segmentation_file)
    raw_segmentation = raw_segmentation.astype(dtype=np.int32)

    binary_segmentation = raw_segmentation > 0

    _debug_cc3d(numpy_array=binary_segmentation, debug=debug)

    ### 1. PRUNE ###
    if prune_threshold > 0:
        pruned_map = cc3d.dust(
            binary_segmentation,
            threshold=prune_threshold,
            connectivity=26,
            in_place=False,
        )
        _debug_cc3d(numpy_array=pruned_map, debug=debug)

        multi_segmentation = raw_segmentation * pruned_map
        binary_segmentation = binary_segmentation * pruned_map
    else:
        multi_segmentation = raw_segmentation

    ### ** Convert to integer ** ###
    multi_segmentation = multi_segmentation.astype(dtype=np.int32)

    ### 2. Connected component analysis ###
    components, N = cc3d.connected_components(binary_segmentation, return_N=True)
    print(N)

    if debug == True:
        print(multi_segmentation.shape)
        print(np.count_nonzero(multi_segmentation))
        pre_uniques, pre_counts = np.unique(multi_segmentation, return_counts=True)

    ### 3. Majority Voting! ###
    if majority_vote == True:
        for segid in range(1, N + 1):
            instance_mask = components == segid
            extracted_image = multi_segmentation * instance_mask
            # majority vote only if the object is not too big
            if np.count_nonzero(extracted_image) < max_mav_threshold:

                # check if circular enough
                circularity = get_circularity(
                    numpy_array=extracted_image.astype(dtype=np.int32), debug=debug
                )
                if circularity > circularity_threshold:

                    # majority vote only if unique most frequent value occurs
                    most_frequent_value = multimode(
                        extracted_image[extracted_image > 0]
                    )
                    # print(len(most_frequent_value))
                    if len(most_frequent_value) == 1:
                        multi_segmentation[extracted_image > 0] = most_frequent_value

            # print(uniques)
            if debug == True:
                print(np.unique(extracted_image))
                print(np.count_nonzero(extracted_image))

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
        # print(multi_segmentation)

    ### ** Save nifti ** ###
    write_nifti(
        numpy_array=multi_segmentation,
        output_nifti_path=polished_segmentation_file,
    )
    lala = read_nifti(polished_segmentation_file)
    print(lala.shape)

    if debug == True:
        print(multi_segmentation.shape)
        print(np.count_nonzero(multi_segmentation))
        uniques, counts = np.unique(multi_segmentation, return_counts=True)

        print("Pre:")
        print(raw_segmentation.shape)
        components, pre_N = cc3d.connected_components(raw_segmentation, return_N=True)
        print(pre_N)
        for uni, count in zip(pre_uniques, pre_counts):
            print("unique:", uni, "// count:", count)

        print("Post:")
        print(multi_segmentation.shape)
        components, post_N = cc3d.connected_components(
            multi_segmentation, return_N=True
        )
        print(post_N)
        for uni, count in zip(uniques, counts):
            print("unique:", uni, "// count:", count)

    return True

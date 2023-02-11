import os
from path import Path
import nibabel as nib
import numpy as np
from tifffile import imread, imwrite


def vprint(*args):
    verbose = False
    # verbose = True
    if verbose:
        print(*args)


def turbopath(the_path):
    return_path = Path(
        os.path.normpath(
            os.path.abspath(
                the_path,
            )
        )
    )
    return return_path


# def read_png(png_path):
#     img_data = imageio.imread(png_path)
#     return img_data


def _read_nifti(nifti_path):
    nifti = nib.load(nifti_path)
    nifti_data = nifti.get_fdata()
    return nifti_data


def _read_tiff(tiff_path):
    data = imread(tiff_path)
    return data


def read_image(image_path):
    format = _get_imageFileFormat(image_path=image_path)

    if format == "tiff":
        data = _read_tiff(tiff_path=image_path)
    elif format == "nifti":
        data = _read_nifti(nifti_path=image_path)

    return data


def _write_nifti(numpy_array, output_nifti_path):
    nifti_image = nib.Nifti1Image(numpy_array, np.eye(4))
    vprint("** saving:", output_nifti_path)
    nib.save(nifti_image, output_nifti_path)
    return True


def _write_tif(numpy_array, output_tiff_path):
    arr = numpy_array.T

    imwrite(output_tiff_path, arr)
    vprint("** saving:", output_tiff_path)
    return True


def _get_imageFileFormat(image_path):
    # TODO consider replacing everything with https://imageio.readthedocs.io/
    tif_ending = image_path[-4:]
    tiff_ending = image_path[-5:]
    nifti_ending = image_path[-7:]

    if tif_ending == ".tif":
        format = "tiff"
    elif tiff_ending == ".tiff":
        format = "tiff"
    elif nifti_ending == ".nii.gz":
        format = "nifti"
    else:
        NotImplementedError(f"this file format is not implemented! // {image_path}")

    return format


def write_image(numpy_array, image_path):
    format = _get_imageFileFormat(image_path=image_path)

    if format == "tiff":
        _write_tif(
            numpy_array=numpy_array,
            output_tiff_path=image_path,
        )
    elif format == "nifti":
        _write_nifti(
            numpy_array=numpy_array,
            output_nifti_path=image_path,
        )

    return True


if __name__ == "__main__":
    data = read_image(
        "/home/florian/flow/EMcapsulins_segmentation/github_EMcapsulins_segmentation/example_data/example_inputs/EM2022_190_K2Box37_03_03_66.tif"
    )
    print(data.shape)
    write_image(
        data,
        "/home/florian/flow/EMcapsulins_segmentation/github_EMcapsulins_segmentation/example_data/lala/test.nii.gz",
    )
    write_image(
        data,
        "/home/florian/flow/EMcapsulins_segmentation/github_EMcapsulins_segmentation/example_data/lala/test.tiff",
    )
    write_image(
        data,
        "/home/florian/flow/EMcapsulins_segmentation/github_EMcapsulins_segmentation/example_data/lala/test.tif",
    )

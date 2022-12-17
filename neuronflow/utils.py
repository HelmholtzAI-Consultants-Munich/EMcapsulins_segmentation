import os
from path import Path
import nibabel as nib
import numpy as np
# import imageio


def vprint(*args):
    verbose = False
    # verbose = True
    if verbose:
        print(*args)


def turbopath(the_path):
    return_path = Path(os.path.normpath(os.path.abspath(the_path)))
    return return_path


# def read_png(png_path):
#     img_data = imageio.imread(png_path)
#     return img_data


def read_nifti(nifti_path):
    nifti = nib.load(nifti_path)
    nifti_data = nifti.get_fdata()
    return nifti_data


def write_nifti(numpy_array, output_nifti_path):
    nifti_image = nib.Nifti1Image(numpy_array, np.eye(4))
    vprint("** saving:", output_nifti_path)
    nib.save(nifti_image, output_nifti_path)
    return True

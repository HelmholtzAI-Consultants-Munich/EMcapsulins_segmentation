import os
from path import Path
import nibabel as nib
import numpy as np
import tifffile
# import imageio


def vprint(*args):
    verbose = False
    # verbose = True
    if verbose:
        print(*args)


def turbopath(the_path):
    return_path = Path(os.path.normpath(os.path.abspath(the_path)))
    return return_path


def read_tif(tif_path):
    tif = tifffile.imread(tif_path)
    return tif

def write_tif(numpy_array, output_tif_path):
    tifffile.imsave(output_tif_path, numpy_array.astype(np.uint8))
    return True


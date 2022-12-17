from tqdm import tqdm

from neuronflow.utils import turbopath
from neuronflow.lib import inference_to_directory



input_folder = turbopath("example_data/example_inputs")
target_folder = turbopath("example_data/example_outputs")

microscopy_images = input_folder.files()  # find all images

# loop through images
for img in tqdm(microscopy_images):
    identity = img.name[:-4]  # image name to use as prefix

    inference_to_directory(
        microscopy_file=img,  # path to the image
        output_folder=target_folder + "/" + identity,  # where to store the outputs
        prefix=identity,  # what identifier to put in front of file names
        cuda_devices="0,1",
    )

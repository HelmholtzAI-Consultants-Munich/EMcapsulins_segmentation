# EMcapsulins segmentation
Deep learning models to segment emcapsulins in 2D TEM micrograph images from the manuscript:

```
Genetically encoded barcodes for correlative volume electron microscopy
```

![Image](documentation/example_pngs/crop.png "cropped microscopy")


## Expected data
The models are trained on 8-bit images with a pixel size: of 0.5525 per nanometer.
We include some example input and output files for replication.

## Installation

1) Clone this repository:
    ```bash
    git clone https://github.com/HelmholtzAI-Consultants-Munich/EMcapsulins_segmentation.git
    ```
2) Go into the repository and install:
    ```
    cd EMcapsulins_segmentation
    pip install -r requirements.txt
    pip install -e .
    ```
    
## Usuage
**run_inference_tiff.py** <-- Example script for inference creating tiff files

**run_inference_nifti.py** <-- Example script for inference creating nifti files


## Citation
when using the software please cite tba

```
tba
```

## Recommended Environment
* CUDA 11.4+
* Python 3.10+
* GPU with at least 8GB of VRAM

further details in requirements.txt

## train your own EMcapsulin segmentation network
Please have a look at this [repository](https://github.com/MartGro/EMcapsulin_Toolbox)

## Licensing

This project is licensed under the terms of the [GNU Affero General Public License v3.0](https://www.gnu.org/licenses/agpl-3.0.de.html).

Contact us regarding licensing.

## Contact / Feedback / Questions
If possible please open a GitHub issue [here](https://github.com/HelmholtzAI-Consultants-Munich/EMcapsulins_segmentation/issues).

For inquiries not suitable for GitHub issues:

felix.sigmund @ tum .de

gil.westmeyer @ tum .de

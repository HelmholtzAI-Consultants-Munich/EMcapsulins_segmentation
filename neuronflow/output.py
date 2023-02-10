import numpy as np
import nibabel as nib
import torch
from PIL import Image
import tifffile as tiff
from neuronflow.utils import turbopath, read_tif, write_tif



def vprint(*args):
    verbose = False
    # verbose = True
    if verbose:
        print(*args)




def _bg_fg_network_output_saver(
    sigmoid_activated_outputs,
    binary_segmentation_file,
    binary_threshold,
    background_output_file,
    foreground_output_file,
):
    bg_data = sigmoid_activated_outputs[0]
    vprint("*** bg_data.shape:", bg_data.shape)
    fg_data = (bg_data * -1) + 1

    if binary_segmentation_file is not None:
        binarized_data = fg_data >= binary_threshold
        binarized_data = binarized_data.astype(dtype=np.uint8)
        write_tif(binarized_data,binary_segmentation_file)
        #tiff.imsave(binary_segmentation_file, binarized_data)

    if background_output_file is not None:
    	write_tif(bg_data.astype(np.uint8),background_output_file)

    if foreground_output_file is not None:
        write_tif(fg_data.astype(np.uint8),foreground_output_file)


def _network_output_saver(
    sigmoid_activated_outputs, channel_number, network_output_file
):
    if network_output_file is not None:
        data = sigmoid_activated_outputs[channel_number]
        vprint("*** data.shape:", data.shape)
        write_tif(data,network_output_file)



def create_output_files(
    onehot_model_outputs_CHW,
    segmentation_file,
    binary_segmentation_file=None,
    binary_threshold=0.5,
    background_output_file=None,
    foreground_output_file=None,
    mQt_output_file=None,
    mmQt_output_file=None,
    mmmQt_output_file=None,
    mMx_output_file=None,
    mmMx_output_file=None,
    mTm_output_file=None,
    mQtTm_output_file=None,
):
    """
    # 1:  "*_elab_1M-Qt.png",
    # 2:  "*_elab_2M-Qt.png",
    # 3:  "*_elab_3M-Qt.png",
    # 4:  "*_elab_1M-Mx.png",
    # 5:  "*_elab_2M-Mx.png",
    # 6:  "*_elab_1M-Tm.png",
    # 7: 1+ 6
    # 8: 1+2+3+4+5+6
    #
    """

    vprint("*** onehot_model_outputs_CHW.shape:", onehot_model_outputs_CHW.shape)

    # generate segmentation
    first_seven_channels = onehot_model_outputs_CHW[0:7]
    vprint("*** first_six_channels.shape:", first_seven_channels.shape)
    segmentation_map = torch.argmax(first_seven_channels, dim=0).detach().cpu().numpy()
    vprint("*** segmentation_map.shape:", segmentation_map.shape)
    segmentation_map_int = segmentation_map.astype(np.uint8)
    vprint("*** segmentation_map_int.shape:", segmentation_map_int.shape)

    write_tif(segmentation_map_int,segmentation_file)
    
    #with open(segmentation_file[:-7]+".npz","wb") as f:
   # 	np.save(segmentation_map_int,f)
    # saving pngs is problematic, therefore we go for nifti
    # cv2.imwrite(output_file, segmentation_map_int)

    # generate model outputs
    sigmoid_activated_outputs = (
        onehot_model_outputs_CHW.sigmoid().detach().cpu().numpy()
    )

    # # TODO https://stackoverflow.com/a/65424772/3485363

    vprint("*** sigmoid_activated_outputs.shape:", sigmoid_activated_outputs.shape)

    # binarized_outputs = sigmoid_activated_outputs >= threshold

    # binarized_outputs = binarized_outputs.astype(np.uint8)

    # vprint("*** binarized_outputs.shape:", binarized_outputs.shape)

    # _network_output_saver(
    #     sigmoid_activated_outputs=sigmoid_activated_outputs,
    #     channel_number=7,
    #     network_output_file=all_output_file,
    # )

    _bg_fg_network_output_saver(
        sigmoid_activated_outputs=sigmoid_activated_outputs,
        binary_segmentation_file=binary_segmentation_file,
        binary_threshold=binary_threshold,
        background_output_file=background_output_file,
        foreground_output_file=foreground_output_file,
    )

    _network_output_saver(
        sigmoid_activated_outputs=sigmoid_activated_outputs,
        channel_number=1,
        network_output_file=mQt_output_file,
    )

    _network_output_saver(
        sigmoid_activated_outputs=sigmoid_activated_outputs,
        channel_number=2,
        network_output_file=mmQt_output_file,
    )

    _network_output_saver(
        sigmoid_activated_outputs=sigmoid_activated_outputs,
        channel_number=3,
        network_output_file=mmmQt_output_file,
    )

    _network_output_saver(
        sigmoid_activated_outputs=sigmoid_activated_outputs,
        channel_number=4,
        network_output_file=mMx_output_file,
    )

    _network_output_saver(
        sigmoid_activated_outputs=sigmoid_activated_outputs,
        channel_number=5,
        network_output_file=mmMx_output_file,
    )

    _network_output_saver(
        sigmoid_activated_outputs=sigmoid_activated_outputs,
        channel_number=6,
        network_output_file=mTm_output_file,
    )

    _network_output_saver(
        sigmoid_activated_outputs=sigmoid_activated_outputs,
        channel_number=7,
        network_output_file=mQtTm_output_file,
    )

"""
Annotate segmentation with text
===============================

Perform a segmentation and annotate the results with
bounding boxes and text

.. tags:: analysis
"""
import numpy as np
import skimage
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops_table
from skimage.morphology import closing, square, remove_small_objects
import napari
from magicgui import magicgui
from magicgui.widgets import FunctionGui
from napari import Viewer
from napari.types import ImageData, LabelsData, LayerDataTuple
  
    
@magicgui(call_button='Save Annotation')
def save_annotation(viewer: Viewer) -> None:
    global label_path
    global label_name
    image = viewer.layers[label_name]
    skimage.io.imsave(label_path,image.data)
    print("Saved Image")
    
    
from PIL import Image
import matplotlib.pyplot as plt

        
micro_image = skimage.io.imread("loading_screen.jpg")
label_image = skimage.io.imread("loading_screen_overlay.png")

def RGB_to_scaled_RGBA(RGB_tuple):
    return (RGB_tuple[0]/255,RGB_tuple[1]/255,RGB_tuple[2]/255,1.0)


color_dict = {
    "1M-Qt":RGB_to_scaled_RGBA((130, 180, 187)),
    "2M-Qt":RGB_to_scaled_RGBA((38, 119, 120)),
    "3M-Qt":RGB_to_scaled_RGBA((37, 94, 121)),
    "1M-Mx":RGB_to_scaled_RGBA((174, 60, 96)),
    "2M-Mx":RGB_to_scaled_RGBA((223, 71, 60)),
    "1M-Tm":RGB_to_scaled_RGBA((243, 195, 60))}

color_dict_with_one_based_index = dict(zip([1,2,3,4,5,6],color_dict.values()))

viewer = napari.view_image(micro_image, name='micro_image', rgb=False)
label_layer = viewer.add_labels(label_image, name='segmentation',color=color_dict_with_one_based_index)


#empty_widget = EmptyGui()
#viewer.window.add_dock_widget(empty_widget, area='right')

##############
#qt Widget
import sys
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5.QtGui import QPixmap

class ImageWidget(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)

    def set_image(self, file_path):
        pixmap = QPixmap(file_path)
        self.setPixmap(pixmap)

image_widget = ImageWidget()
image_widget.set_image('colormap_white.png')


###################



###
#file viewer widget 

import os

import pathlib
from pathlib import Path



base_path = None

@magicgui(fn={'mode': 'd'}, call_button='Scan Directory')
#def path_widget(fn =  pathlib.Path.home()):
def path_widget(fn =  Path(os.getcwd())):
    print(fn)
    global base_path
    base_path = fn
    list_images(base_path)
    
#########

import os
import glob



#@magicgui(call_button='Scan Files')
def list_images(path)->list:
    """
    A magicgui widget that takes a path and lists all .png images in that path
    """
    print("Scanning Directory...")
    #path = base_path
    #png_files = [entry.name for entry in os.scandir(path) if entry.name.endswith('.png') and entry.is_file()]

    def scantree(path):
        """Recursively yield DirEntry objects for given directory."""
        for entry in os.scandir(path):
            if entry.is_dir(follow_symlinks=False):
                yield from scantree(entry.path)  # see below for Python 2.x
            else:
                yield entry
    
    global png_files
    
    png_files = {}
    
    for entry in scantree(base_path):
        if entry.name.endswith('_mic.tif') and entry.is_file():
            png_files[entry.name] = entry.path
    
    
    png_files = {key:png_files[key] for key in sorted(png_files.keys())}
    
    print(png_files)
    
    global selector
    selector.x.choices = png_files.copy().keys()
    #selector.update_args()

    return png_files


####

global png_files

png_files = {}

micro_path = None
label_path = None
label_name = None


@magicgui(x=dict(widget_type='Select', choices=png_files.keys()))
def selector(x):
    update = False
    
    png_key = x[0]
    
    global png_files 
        
    print("PNG KEYS: ")
    print(png_files.keys())
    
    if png_key is not None:
        global micro_path
        global label_path
        micro_path = png_files[png_key]
        label_path = micro_path.rstrip("_mic.tif")+"_label.tif"
        new_micro_image = skimage.io.imread(micro_path)
        try:
            new_label_image = skimage.io.imread(label_path)
        except:
            print("no label found")
        update = True
    
    
    if update:
        global viewer 
        try:
            viewer.layers.remove(viewer.layers[0])
        except:
            print("could not remove layers")
            
        
        try:
            viewer.layers.remove(viewer.layers[0])
        except:
            print("could not remove layers")
        
        
        
        global label_name
        label_name = 'seg:'+str(label_path.split("/")[-1])
        viewer.add_image(new_micro_image,name='mic:'+str(micro_path).split("/")[-1])
        try: 
            label_layer = viewer.add_labels(new_label_image, name=label_name,color=color_dict_with_one_based_index)
        except:
            pass
    
        selector.x.choices = png_files.copy().keys()


    print(x)
    
    
    #viewer = napari.view_image(new_micro_image, name='micro_image', rgb=False)
    #label_layer = viewer.add_labels(new_label_image, name='segmentation',color=color_dict_with_one_based_index)


#@magicgui(call_button='call')
##def path_widget(fn =  pathlib.Path.home()):
#def empty_widget():
#    print("hit")
#    def path_widget(fn =  Path(os.getcwd())):
#        print(fn)
#        global base_path
#        base_path = fn





save_annotation_widget = viewer.window.add_dock_widget(save_annotation)
save_annotation_widget.name = "Save Annotation"

#viewer.window.add_dock_widget(display_image)
color_widget = viewer.window.add_dock_widget(image_widget,area="left")
color_widget.name = "Color Reference"

path_selector = viewer.window.add_dock_widget(path_widget,area="right")
path_selector.name = "Path Selector"
#viewer.window.add_dock_widget(list_images,area="right")

file_display_widget = viewer.window.add_dock_widget(selector,area="right")
file_display_widget.setFixedHeight(650)
file_display_widget.name = "Image Selector"
selector.x.min_height = 600

#list_view = selector.x.widget.list_view
#list_view.setFixedHeight(400)






import code
code.InteractiveConsole(locals=globals()).interact()

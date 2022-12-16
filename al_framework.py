import sys
import os
from pathlib import Path
from typing import List
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMainWindow, QFileSystemModel, QListView, QHBoxLayout, QFileIconProvider, QLabel, QFileDialog, QLineEdit, QTreeView
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon
import napari
import torch 
from cellpose import models
from utils import merge_2d_labels, make_bbox
from skimage.measure import regionprops_table
import warnings
warnings.simplefilter('ignore')

ICON_SIZE = QSize(512,512)
accepted_types = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
# Apply threshold on GFP
threshold_gfp = 300 # Define the threshold looking at the plots

def changeWindow(w1, w2):
    w1.hide()
    w2.show()

class IconProvider(QFileIconProvider):

    def __init__(self) -> None:
        super().__init__()

    def icon(self, type: 'QFileIconProvider.IconType'):

        fn = type.filePath()

        if fn.endswith(accepted_types):
            a = QPixmap(ICON_SIZE)
            a.load(fn)
            return QIcon(a)
        else:
            return super().icon(type)

class NapariWindow(QWidget):
    def __init__(self, 
                img_filename_dapi,
                img_filename_gfp,
                eval_data_path,
                train_data_path):
        super().__init__()

        self.img_filename_dapi = img_filename_dapi
        self.img_filename_gfp = img_filename_gfp
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path

        self.root_name = self.img_filename_dapi.split('DAPI Channel')[0]
        potential_seg_name = self.root_name + '_seg.tiff' #Path(img_filename).stem+'_seg.tiff' #+Path(img_filename).suffix
        potential_class_name = self.root_name + '_classes.tiff' #Path(img_filename).stem+'_classes.tiff' #+Path(img_filename).suffix

        self.setWindowTitle("napari Viewer")
        in_eval = False
        # check if the image is in the uncurated folder
        if os.path.exists(os.path.join(self.eval_data_path, self.img_filename_dapi)):
            in_eval = True
            self.img = imread(os.path.join(self.eval_data_path, self.img_filename_dapi))
            # check if object segmentation and class masks are in the folder and read them if they are there  
            if os.path.exists(os.path.join(self.eval_data_path, potential_seg_name)):
                seg = imread(os.path.join(self.eval_data_path, potential_seg_name))
            else: seg = None
            if os.path.exists(os.path.join(self.eval_data_path, potential_class_name)):
                classes = imread(os.path.join(self.eval_data_path, potential_class_name))
            else: classes = None

        # if not it will be in the curated folder
        else: 
            self.img = imread(os.path.join(self.train_data_path, self.img_filename_dapi))
            # check if object segmentation and class masks are in the folder and read them if they are there 
            if os.path.exists(os.path.join(self.train_data_path, potential_seg_name)):
                seg = imread(os.path.join(self.train_data_path, potential_seg_name))
            else: seg = None
            if os.path.exists(os.path.join(self.train_data_path, potential_class_name)):
                classes = imread(os.path.join(self.train_data_path, potential_class_name))
            else: classes = None

        # check and read gfp image too
        if os.path.exists(os.path.join(self.eval_data_path, self.img_filename_gfp)):
            self.img_gfp = imread(os.path.join(self.eval_data_path, self.img_filename_gfp))
        elif os.path.exists(os.path.join(self.train_data_path, self.img_filename_gfp)):
            self.img_gfp = imread(os.path.join(self.train_data_path, self.img_filename_gfp))
        else:
            self.img_gfp = None
        
        # start napari and load all data into the viewer
        self.viewer = napari.Viewer(show=False)
        self.viewer.add_image(self.img, name='DAPI Channel')
        
        if self.img_gfp is not None:
            self.viewer.add_image(self.img_gfp, name='GFP Channel')

        if seg is not None: 
            self.viewer.add_labels(seg, name='Cell Objects')

        if classes is not None:
            self.viewer.add_labels(classes, name='Cell Classes')

        # add napari viewer to the window
        main_window = self.viewer.window._qt_window
        layout = QVBoxLayout()
        layout.addWidget(main_window)

        add_button = QPushButton('Add to training data')
        layout.addWidget(add_button)
        add_button.clicked.connect(self.on_add_button_clicked)
        if not in_eval: add_button.hide()

        # this is commented because it is the same as closing the window
        #self.return_button = QPushButton('Return')
        #layout.addWidget(self.return_button)
        #self.return_button.clicked.connect(self.on_return_button_clicked)

        self.setLayout(layout)
        self.show()

    def _get_layer_names(self, layer_type: napari.layers.Layer = napari.layers.Labels) -> List[str]:
        """
        Get list of layer names of a given layer type.
        """
        layer_names = [
            layer.name
            for layer in self.viewer.layers
            if type(layer) == layer_type
        ]
        if layer_names:
            return [] + layer_names
        else:
            return []

    def on_add_button_clicked(self):

        # move the image to the curated dataset folder
        os.replace(os.path.join(self.eval_data_path, self.img_filename_dapi), os.path.join(self.train_data_path, self.img_filename_dapi))
        # if we have a gfp image then move that too to the curated dataset folder
        if os.path.exists(os.path.join(self.eval_data_path, self.img_filename_gfp)):
            os.replace(os.path.join(self.eval_data_path, self.img_filename_gfp), os.path.join(self.train_data_path, self.img_filename_gfp))

        # get the napari layer names
        label_names = self._get_layer_names()
        undefined_layers = []
        seg = None
        classes = None
        # save the labels - ideally we have labels with names 'Cell Objects' and 'Cell Classes'
        for name in label_names:
            if 'Objects' in name:
                seg = self.viewer.layers[name].data
                seg_name = self.root_name+'_seg.tiff' #+Path(self.img_filename_dapi).suffix
                imsave(os.path.join(self.train_data_path, seg_name),seg)
                if os.path.exists(os.path.join(self.eval_data_path, seg_name)): 
                    os.remove(os.path.join(self.eval_data_path, seg_name))
            elif 'Classes' in name:
                classes = self.viewer.layers[name].data
                classes_name = self.root_name+'_classes.tiff' #+Path(self.img_filename_dapi).suffix
                imsave(os.path.join(self.train_data_path, classes_name), classes)
                if os.path.exists(os.path.join(self.eval_data_path, classes_name)): 
                    os.remove(os.path.join(self.eval_data_path, classes_name))   
            else:
                undefined_layers.append(self.viewer.layers[name].data)

        if len(undefined_layers)!= 0:
            for idx, label in enumerate(undefined_layers):
                name= 'label'+str(idx)+'.tif'
                imsave(os.path.join(self.train_data_path, name), label)
                print('Warning: you have created file with name: ', name,'. Please rename this with an extension _seg or _classes if you would like to use it within this tool again.')

        self.close()

    '''
    def on_return_button_clicked(self):
        self.close()
    '''

class MainWindow(QWidget):
    def __init__(self, eval_data_path, train_data_path):
        super().__init__()

        self.title = "Data Overview"
        self.eval_data_path = eval_data_path
        self.train_data_path = train_data_path
        self.main_window()

    def main_window(self):
        self.setWindowTitle(self.title)
        #self.resize(1000, 1500)
        self.main_layout = QVBoxLayout()  
        self.top_layout = QHBoxLayout()
        self.bottom_layout = QHBoxLayout()

        self.eval_dir_layout = QVBoxLayout() 
        self.eval_dir_layout.setContentsMargins(0,0,0,0)
        self.label_eval = QLabel(self)
        self.label_eval.setText("Uncurated dataset")
        self.eval_dir_layout.addWidget(self.label_eval)
        # add eval dir list
        model_eval = QFileSystemModel()
        model_eval.setIconProvider(IconProvider())
        self.list_view_eval = QTreeView(self)
        self.list_view_eval.setModel(model_eval)
        for i in range(1,4):
            self.list_view_eval.hideColumn(i)
        #self.list_view_eval.setFixedSize(600, 600)
        self.list_view_eval.setRootIndex(model_eval.setRootPath(self.eval_data_path)) 
        self.list_view_eval.clicked.connect(self.item_eval_selected)
        self.cur_selected_img = None
        self.eval_dir_layout.addWidget(self.list_view_eval)
        self.top_layout.addLayout(self.eval_dir_layout)

        self.train_dir_layout = QVBoxLayout() 
        self.train_dir_layout.setContentsMargins(0,0,0,0)
        self.label_train = QLabel(self)
        self.label_train.setText("Curated dataset")
        self.train_dir_layout.addWidget(self.label_train)
        # add train dir list
        model_train = QFileSystemModel()
        #self.list_view = QListView(self)
        self.list_view_train = QTreeView(self)
        model_train.setIconProvider(IconProvider())
        self.list_view_train.setModel(model_train)
        for i in range(1,4):
            self.list_view_train.hideColumn(i)
        #self.list_view_train.setFixedSize(600, 600)
        self.list_view_train.setRootIndex(model_train.setRootPath(self.train_data_path)) 
        self.list_view_train.clicked.connect(self.item_train_selected)
        self.train_dir_layout.addWidget(self.list_view_train)
        self.top_layout.addLayout(self.train_dir_layout)

        self.main_layout.addLayout(self.top_layout)
        
        # add buttons
        self.launch_nap_button = QPushButton("View image and fix label", self)
        self.launch_nap_button.clicked.connect(self.launch_napari_window)  # add selected image    
        self.bottom_layout.addWidget(self.launch_nap_button)
        
        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)  # add selected image    
        self.bottom_layout.addWidget(self.train_button)

        self.inference_button = QPushButton("Generate Labels", self)
        self.inference_button.clicked.connect(self.run_inference)  # add selected image    
        self.bottom_layout.addWidget(self.inference_button)

        self.main_layout.addLayout(self.bottom_layout)

        self.setLayout(self.main_layout)
        self.show()

    def launch_napari_window(self):
        
        if os.path.exists(os.path.join(self.eval_data_path, self.cur_selected_img)):
            in_eval = True
        else: in_eval = False

        if 'GFP Channel' in self.cur_selected_img:
            gfp_img = self.cur_selected_img
            name_split = gfp_img.split('GFP Channel')[0]
            if in_eval:
                dapi_img = [file for file in os.listdir(self.eval_data_path) if name_split in file and 'DAPI Channel' in file][0]
            else:
                dapi_img = [file for file in os.listdir(self.train_data_path) if name_split in file and 'DAPI Channel' in file][0]

        else: 
            dapi_img = self.cur_selected_img
            name_split = dapi_img.split('DAPI Channel')[0]
            if in_eval:
                gfp_img = [file for file in os.listdir(self.eval_data_path) if name_split in file and 'GFP Channel' in file][0]
            else:
                gfp_img = [file for file in os.listdir(self.train_data_path) if name_split in file and 'GFP Channel' in file][0]
        self.nap_win = NapariWindow(img_filename_dapi=dapi_img,
                                    img_filename_gfp=gfp_img,
                                    eval_data_path=self.eval_data_path, 
                                    train_data_path=self.train_data_path)
        self.nap_win.show()

    def item_eval_selected(self, item):
        self.cur_selected_img = item.data()
    
    def item_train_selected(self, item):
        self.cur_selected_img = item.data()

    def train_model(self):
        pass

    def run_inference(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device=="cuda":
            model = models.Cellpose(gpu=True, model_type="cyto")
        else:
            model = models.Cellpose(gpu=False, model_type="cyto")
        
        list_files_dapi = [file for file in os.listdir(self.eval_data_path) if Path(file).suffix in accepted_types and 'GFP Channel' not in file]
        list_files_gfp = [file for file in os.listdir(self.eval_data_path) if Path(file).suffix in accepted_types and 'GFP Channel' in file]
        list_files_dapi.sort()
        list_files_gfp.sort()

        for idx, img_filename in enumerate(list_files_dapi):
            # don't do this for segmentations in the folder
            #extend to check the prefix also matches an existing image
            #seg_name = Path(self.img_filename).stem+'_seg'+Path(self.img_filename).suffix
            if '_seg' in img_filename or '_classes' in img_filename:  continue

            else:
                img = imread(os.path.join(self.eval_data_path, img_filename)) #DAPI
                img_gfp = imread(os.path.join(self.eval_data_path, list_files_gfp[idx]))
                orig_size = img.shape
                name_split = img_filename.split('DAPI Channel')[0]
                seg_name = name_split + '_seg.tiff' #Path(img_filename).stem+'_seg.tiff' #+Path(img_filename).suffix
                class_name = name_split + '_classes.tiff' #Path(img_filename).stem+'_classes.tiff' #+Path(img_filename).suffix

                if Path(img_filename).suffix in (".tiff", ".tif") and len(orig_size)==3:

                    print('Warning: 3D image stack found. We are assuming your first dimension is your stack dimension. Please cross check this.')
                    height, width = orig_size[1], orig_size[2]
                    
                    max_dim = max(height, width)
                    rescale_factor = max_dim/512
                    img = rescale(img, 1/rescale_factor, channel_axis=0)
                    mask, _, _, _ = model.eval(img, z_axis=0) #for 3D
                    mask = merge_2d_labels(mask)
                    mask = resize(mask, (orig_size[0], height, width), order=0)
                    
                    # get labels of object segmentation
                    labels = np.unique(mask)[1:]
                    class_mask = np.copy(mask)
                    class_mask[mask != 0] = 1 # set all objects to class 1
                    
                    # if the mean of an object in the gfp image is abpve the predefined threshold set it to class 2
                    for l in labels:
                        mean_l = np.mean(img_gfp[mask == l])
                        if mean_l > threshold_gfp:
                            class_mask[mask == l] = 2                   
                    
                else: 
                    print('Image type not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale. \
                            Currently supported image file formats are: ', accepted_types, 'Exiting now.')
                    sys.exit()

                imsave(os.path.join(self.eval_data_path, seg_name), mask)
                imsave(os.path.join(self.eval_data_path, class_name), class_mask)


class WelcomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(200, 200)
        self.title = "Select Dataset"
        self.main_layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setText('Welcome to Helmholtz AI data centric tool! Please select your dataset folder')
        
        self.val_layout = QHBoxLayout()
        self.val_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_eval_clicked)
        self.val_layout.addWidget(self.val_textbox)
        self.val_layout.addWidget(self.fileOpenButton)

        self.train_layout = QHBoxLayout()
        self.train_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_train_clicked)
        self.train_layout.addWidget(self.train_textbox)
        self.train_layout.addWidget(self.fileOpenButton)

        self.main_layout.addWidget(self.label)
        self.main_layout.addLayout(self.val_layout)
        self.main_layout.addLayout(self.train_layout)

        self.start_button = QPushButton('Start', self)
        self.start_button.show()
        self.start_button.clicked.connect(self.start_main)
        self.main_layout.addWidget(self.start_button)
        self.setLayout(self.main_layout)
        self.show()

    def browse_eval_clicked(self):
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.filename_val = fd.selectedFiles()[0]
        self.val_textbox.setText(self.filename_val)
    
    def browse_train_clicked(self):
        fd = QFileDialog()
        fd.setFileMode(QFileDialog.Directory)
        if fd.exec_():
            self.filename_train = fd.selectedFiles()[0]
        self.train_textbox.setText(self.filename_train)

    
    def start_main(self):
        self.hide()
        self.mw = MainWindow(self.filename_val, self.filename_train)
    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WelcomeWindow()
    # change to window = MainWindow('uncurated/data/path','curated/data/path') if you want to skip first window
    sys.exit(app.exec())

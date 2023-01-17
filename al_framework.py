import sys
import os

from typing import List
import numpy as np
from skimage.io import imsave
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QMainWindow, QFileSystemModel, QListView, QHBoxLayout, QFileIconProvider, QLabel, QFileDialog, QLineEdit, QTreeView, QMessageBox
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon
import napari

from utils import get_predictions, get_channel_files, read_files

ICON_SIZE = QSize(512,512)
accepted_types = (".jpg", ".jpeg", ".png", ".tiff", ".tif")
# Apply threshold on GFP
threshold_gfp = 300 # Define the threshold looking at the plots

class IconProvider(QFileIconProvider):
    def __init__(self) -> None:
        super().__init__()

    def icon(self, type: 'QFileIconProvider.IconType'):
        try:
            fn = type.filePath()

            if fn.endswith(accepted_types):
                a = QPixmap(ICON_SIZE)
                a.load(fn)
                return QIcon(a)
            else:
                return super().icon(type)
        except AttributeError: return super().icon(type)


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

        self.setWindowTitle("napari Viewer")
        in_eval = False
        img, img_gfp, self.seg, self.classes, in_eval, self.root_name = read_files(self.eval_data_path, self.train_data_path, self.img_filename_dapi, self.img_filename_gfp)
        # start napari and load all data into the viewer
        self.viewer = napari.Viewer(show=False)
        self.viewer.add_image(img, name='DAPI Channel')
        self.viewer.add_image(img_gfp, name='GFP Channel')
        if self.seg is not None: 
            self.viewer.add_labels(self.seg, name='Cell Objects')
        if self.classes is not None:
            self.viewer.add_labels(self.classes, name='Cell Classes')

        # add napari viewer to the window
        main_window = self.viewer.window._qt_window
        layout = QVBoxLayout()
        layout.addWidget(main_window)
        '''
        # this is currently not working. Problem with napari?
        if self.seg is not None or self.classes is not None:
            reset_button = QPushButton('Reset masks')
            layout.addWidget(reset_button)
            reset_button.clicked.connect(self.on_reset_button_clicked)
        '''
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
    '''
    def on_reset_button_clicked(self):
        if self.seg is not None: 
            self.viewer.layers['Cell Objects'].data = self.seg
            self.viewer.layers['Cell Objects'].refresh()
        if self.classes is not None:
            self.viewer.layers['Cell Classes'].data = self.classes
            self.viewer.layers['Cell Classes'].refresh()
    '''

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
                name= 'label'+str(idx)+'.tiff'
                imsave(os.path.join(self.train_data_path, name), label)
                txt = 'Warning: you have created file with name: '+name+'. \
                        Please rename this to '+self.root_name+'_classes.tiff or '+self.root_name+'_seg.tiff if \
                        you would like to use it within this tool again.'
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Information)
                msg.setText(txt)
                msg.setWindowTitle("Warning")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec()
                
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
        self.launch_nap_button = QPushButton("View image and fix labels", self)
        self.launch_nap_button.clicked.connect(self.launch_napari_window)  # add selected image    
        self.bottom_layout.addWidget(self.launch_nap_button)
        '''
        self.train_button = QPushButton("Train Model", self)
        self.train_button.clicked.connect(self.train_model)  # add selected image    
        self.bottom_layout.addWidget(self.train_button)
        '''
        self.inference_button = QPushButton("Generate Labels", self)
        self.inference_button.clicked.connect(self.run_inference)  # add selected image    
        self.bottom_layout.addWidget(self.inference_button)

        self.main_layout.addLayout(self.bottom_layout)

        self.setLayout(self.main_layout)
        self.show()

    def launch_napari_window(self):
        if not self.cur_selected_img or '_seg.tiff' in self.cur_selected_img or '_classes.tiff' in self.cur_selected_img:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Please first select an image you wish to visualise. The selected image must belong be an original images, not a mask.")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        else:
            dapi_img, gfp_img = get_channel_files(self.eval_data_path, self.train_data_path, self.cur_selected_img)
            self.nap_win = NapariWindow(img_filename_dapi=dapi_img,
                                        img_filename_gfp=gfp_img,
                                        eval_data_path=self.eval_data_path, 
                                        train_data_path=self.train_data_path)
            self.nap_win.show()

    def item_eval_selected(self, item):
        self.cur_selected_img = item.data()
    
    def item_train_selected(self, item):
        self.cur_selected_img = item.data()
    '''
    def train_model(self):
        pass
    '''
    def run_inference(self):
        warn_flag = get_predictions(self.eval_data_path, threshold_gfp, accepted_types)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        if warn_flag == '3d':
            msg.setText("Warning: A 3D image stack was found. We are assuming your first dimension is your stack dimension.  \
                        Please confirm this is the case or change your data type if not.")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
        else:
            text = "Image type not supported. Only 2D and 3D image shapes currently supported. 3D stacks must be of type grayscale with z axis in first dimension. \
                        Currently supported image file formats are: \n"+ str(accepted_types)+"\n Closing program now."
            msg.setText(text)
            msg.setWindowTitle("Error")
            msg.setStandardButtons(QMessageBox.Ok)
            r = msg.exec()
            self.hide()
            self.close()

class WelcomeWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(200, 200)
        self.title = "Select Dataset"
        self.main_layout = QVBoxLayout()
        self.label = QLabel(self)
        self.label.setText('Welcome to Helmholtz AI data centric tool! Please select your dataset folders')
        
        self.val_layout = QHBoxLayout()
        val_label = QLabel(self)
        val_label.setText('Uncurated dataset path:')
        self.val_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_eval_clicked)
        self.val_layout.addWidget(val_label)
        self.val_layout.addWidget(self.val_textbox)
        self.val_layout.addWidget(self.fileOpenButton)

        self.train_layout = QHBoxLayout()
        train_label = QLabel(self)
        train_label.setText('Curated dataset path:')
        self.train_textbox = QLineEdit(self)
        self.fileOpenButton = QPushButton('Browse',self)
        self.fileOpenButton.show()
        self.fileOpenButton.clicked.connect(self.browse_train_clicked)
        self.train_layout.addWidget(train_label)
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

        self.filename_train = ''
        self.filename_val = ''

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
        if self.filename_train and self.filename_val:
            self.hide()
            self.mw = MainWindow(self.filename_val, self.filename_train)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("You need to specify a folder both for your uncurated and curated dataset (even if the curated folder is currently empty). Please go back and select folders for both.")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()

    

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WelcomeWindow()
    # change to window = MainWindow('uncurated/data/path','curated/data/path') if you want to skip first window
    sys.exit(app.exec())

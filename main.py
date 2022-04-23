from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QCheckBox, QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QSlider, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
import sys

import cv2
import numpy as np
from FaceModel import FaceModel
from PatchModel import PatchModel

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Principal Components demo :D")
        self.face_model = FaceModel('landmark_data',10)
        self.disply_width = 640
        self.display_height = 480
        self.scale = 250
        self.x_ofs = 250
        self.y_ofs = 250
        self.pc_weight = np.zeros(self.face_model.n_pc)
        self.is_background = False
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.patch_label = QLabel(self)
        #self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('lena.png')

        self.grid = QGridLayout()

        self.backgound_checkbox = QCheckBox()
        self.backgound_checkbox.setText('use background')
        self.backgound_checkbox.stateChanged.connect(self.on_check_changed)

        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setMinimum(0)
        self.slider_scale.setMaximum(500)
        self.slider_scale.setValue(250)
        self.slider_scale.valueChanged.connect(self.on_scale_change)

        self.slider_x_ofs = QSlider(Qt.Horizontal)
        self.slider_x_ofs.setMinimum(0)
        self.slider_x_ofs.setMaximum(500)
        self.slider_x_ofs.setValue(250)
        self.slider_x_ofs.valueChanged.connect(self.on_ofs_change)

        self.slider_y_ofs = QSlider(Qt.Horizontal)
        self.slider_y_ofs.setMinimum(0)
        self.slider_y_ofs.setMaximum(500)
        self.slider_y_ofs.setValue(250)
        self.slider_y_ofs.valueChanged.connect(self.on_ofs_change)

        self.sliders_pc = []
        for i in range(0, self.face_model.n_pc):
            self.sliders_pc.append(QSlider(Qt.Horizontal))
            self.sliders_pc[i].setMinimum(-100)
            self.sliders_pc[i].setMaximum(100)
            self.sliders_pc[i].setValue(0)
            self.sliders_pc[i].valueChanged.connect(self.on_pc_changed)

        # create a vertical box layout and add the two labels
        self.grid.addWidget(self.image_label,0, 0)
        self.grid.addWidget(self.patch_label,0, 1)
        vbox = QVBoxLayout()
        # vbox.addWidget(self.image_label)
        vbox.addWidget(self.backgound_checkbox)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(QLabel('scale'))
        vbox.addWidget(self.slider_scale)
        vbox.addWidget(QLabel('x-offset'))
        vbox.addWidget(self.slider_x_ofs)
        vbox.addWidget(QLabel('y-offset'))
        vbox.addWidget(self.slider_y_ofs)
        for i in range(0, self.face_model.n_pc):
            vbox.addWidget(QLabel(str(i+1)+'-PC'))
            vbox.addWidget(self.sliders_pc[i])
        self.grid.addLayout(vbox,1,0)
        self.setLayout(self.grid)
        self.lena = cv2.imread('lena.png')
        self.background = np.zeros((self.lena.shape[0], self.lena.shape[1], 3), np.uint8)
        self.refresh_image()
        self.load_image()

        # patch
        self.patch_model = PatchModel()
        self.draw_image_sub(self.patch_model.extract_patch()[0])
        print(type(self.patch_model._weight_image))
        self.draw_image_sub(
            cv2.cvtColor(self.patch_model._weight_image, cv2.COLOR_GRAY2BGR))


    def refresh_image(self):
        ret = self.render_face(self.lena if self.is_background else self.background)
        qt_img = self.convert_cv_qt(ret)
        self.image_label.setPixmap(qt_img)

    def on_check_changed(self, state):
        self.is_background = state
        self.refresh_image()
    def on_scale_change(self, value):
        self.scale = value
        self.refresh_image()
        
    def on_pc_changed(self,value):
        for i in range(0, self.face_model.n_pc):
            self.pc_weight[i] = self.sliders_pc[i].value()/100.0
        self.refresh_image()

    def load_image(self, filename='indoor_001.png'):
        self.img = cv2.imread(filename)
        self.draw_image(self.plot_pts(self.img, self.load_pts()))
        pass

    def draw_image_sub(self, img):
        self.img = img
        qt_img = self.convert_cv_qt(self.img)
        self.patch_label.setPixmap(qt_img)

    def draw_image(self, img):
        self.img = img
        qt_img = self.convert_cv_qt(self.img)
        self.image_label.setPixmap(qt_img)

    def load_pts(self, filename='indoor_001.pts'):
        data = self.face_model.read_pts_file(filename)
        return data
    def plot_pts(self, src, data):
        for d in data:
            x = int(d[0])
            y = int(d[1])
            src = cv2.circle(img=src, center = (x,y), radius =3, color =(0,255,0), thickness=-1)
        return src

    def render_face(self, cv_img):
        img = cv_img.copy()
        #TODO: move this out!
        weighted_sum = np.zeros((68,2))
        for i in range(0, self.face_model.n_pc):
            weighted_sum  = weighted_sum + self.face_model.pc[i] * self.pc_weight[i]
        data = self.face_model.mu+weighted_sum
        for d in data:
            x = int(d[0]*self.scale)+self.x_ofs
            y = int(d[1]*self.scale)+self.y_ofs
            img = cv2.circle(img=img, center = (x,y), radius =3, color =(0,255,0), thickness=-1)
        return img
    
    def on_ofs_change(self, value):
        self.x_ofs = self.slider_x_ofs.value()
        self.y_ofs = self.slider_y_ofs.value()
        self.refresh_image()


    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
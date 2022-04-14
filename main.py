from pydoc import render_doc
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout, QSlider
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
import sys
import cv2
from matplotlib import scale
import numpy as np
from pts_parse import get_mean_face, read_pts_file, move_center, normalize_face, get_principal_components


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Principal Components demo :D")
        self.disply_width = 640
        self.display_height = 480
        self.scale = 250
        self.x_ofs = 250
        self.y_ofs = 250
        self.pc_weight = [0,0,0]
        self.mean_face = get_mean_face()
        self.pc = get_principal_components()
        # create the label that holds the image
        self.image_label = QLabel(self)
        #self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('lena.png')
        self.slider_scale = QSlider(Qt.Horizontal)
        self.slider_scale.setMinimum(0)
        self.slider_scale.setMaximum(500)
        self.slider_scale.setValue(250)
        self.slider_scale.valueChanged.connect(self.on_scale_change)

        self.slider_x_ofs = QSlider(Qt.Horizontal)
        self.slider_x_ofs.setMinimum(0)
        self.slider_x_ofs.setMaximum(500)
        self.slider_x_ofs.setValue(250)
        self.slider_x_ofs.valueChanged.connect(self.on_xofs_change)

        self.slider_y_ofs = QSlider(Qt.Horizontal)
        self.slider_y_ofs.setMinimum(0)
        self.slider_y_ofs.setMaximum(500)
        self.slider_y_ofs.setValue(250)
        self.slider_y_ofs.valueChanged.connect(self.on_yofs_change)

        self.slider_pc_1 = QSlider(Qt.Horizontal)
        self.slider_pc_1.setMinimum(-100)
        self.slider_pc_1.setMaximum(100)
        self.slider_pc_1.setValue(0)
        self.slider_pc_1.valueChanged.connect(self.on_pc_changed)

        self.slider_pc_2 = QSlider(Qt.Horizontal)
        self.slider_pc_2.setMinimum(-100)
        self.slider_pc_2.setMaximum(100)
        self.slider_pc_2.setValue(0)
        self.slider_pc_2.valueChanged.connect(self.on_pc_changed)

        self.slider_pc_3 = QSlider(Qt.Horizontal)
        self.slider_pc_3.setMinimum(-100)
        self.slider_pc_3.setMaximum(100)
        self.slider_pc_3.setValue(0)
        self.slider_pc_3.valueChanged.connect(self.on_pc_changed)

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        vbox.addWidget(QLabel('scale'))
        vbox.addWidget(self.slider_scale)
        vbox.addWidget(QLabel('x-offset'))
        vbox.addWidget(self.slider_x_ofs)
        vbox.addWidget(QLabel('y-offset'))
        vbox.addWidget(self.slider_y_ofs)
        vbox.addWidget(QLabel('1st-PC'))
        vbox.addWidget(self.slider_pc_1)
        vbox.addWidget(QLabel('2nd-PC'))
        vbox.addWidget(self.slider_pc_2)
        vbox.addWidget(QLabel('3rd-PC'))
        vbox.addWidget(self.slider_pc_3)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
        # don't need the grey image now
        #grey = QPixmap(self.disply_width, self.display_height)
        #grey.fill(QColor('darkGray'))
        #self.image_label.setPixmap(grey)

        # load the test image - we really should have checked that this worked!
        self.lena = cv2.imread('lena.png')
        self.background = np.zeros((self.lena.shape[0], self.lena.shape[1], 3), np.uint8)
        ret = self.render_image(self.lena)
        # convert the image to Qt format
        qt_img = self.convert_cv_qt(ret)
        # display it
        self.image_label.setPixmap(qt_img)

    def on_scale_change(self, value):
        self.scale = value
        ret = self.render_image(self.lena)
        qt_img = self.convert_cv_qt(ret)
        self.image_label.setPixmap(qt_img)
        
    def on_pc_changed(self,value):
        self.pc_weight[0] = self.slider_pc_1.value()/100.0
        self.pc_weight[1] = self.slider_pc_2.value()/100.0
        self.pc_weight[2] = self.slider_pc_3.value()/100.0
        ret = self.render_image(self.lena)
        qt_img = self.convert_cv_qt(ret)
        self.image_label.setPixmap(qt_img)

    def render_image(self, cv_img):
        # cv2.circle(img=cv_img, center = (250,250), radius =10, color =(0,255,0), thickness=-1)
        # data = read_pts_file()
        # data = move_center(data)
        # data = normalize_data(data)
        img = cv_img.copy()
        #TODO: move this out!
        data = self.mean_face+self.pc[0] * self.pc_weight[0]+self.pc[1] * self.pc_weight[1]+self.pc[2] * self.pc_weight[2]
        for d in data:
            x = int(d[0]*self.scale)+self.x_ofs
            y = int(d[1]*self.scale)+self.y_ofs
            img = cv2.circle(img=img, center = (x,y), radius =3, color =(0,255,0), thickness=-1)
        return img
    
    def on_xofs_change(self, value):
        self.x_ofs = value
        ret = self.render_image(self.lena)
        qt_img = self.convert_cv_qt(ret)
        self.image_label.setPixmap(qt_img)

    def on_yofs_change(self, value):
        self.y_ofs = value
        ret = self.render_image(self.lena)
        qt_img = self.convert_cv_qt(ret)
        self.image_label.setPixmap(qt_img)

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
from pydoc import render_doc
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt,QObject
import sys
import cv2

from pts_parse import read_pts_file, move_center, normalize_data


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Qt static label demo")
        self.disply_width = 640
        self.display_height = 480
        # create the label that holds the image
        self.image_label = QLabel(self)
        #self.image_label.resize(self.disply_width, self.display_height)
        # create a text label
        self.textLabel = QLabel('lena.png')

        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)
        # don't need the grey image now
        #grey = QPixmap(self.disply_width, self.display_height)
        #grey.fill(QColor('darkGray'))
        #self.image_label.setPixmap(grey)

        # load the test image - we really should have checked that this worked!
        cv_img = cv2.imread('lena.png')
        self.render_image(cv_img)
        # convert the image to Qt format
        qt_img = self.convert_cv_qt(cv_img)
        # display it
        self.image_label.setPixmap(qt_img)

    def render_image(self, cv_img):
        # cv2.circle(img=cv_img, center = (250,250), radius =10, color =(0,255,0), thickness=-1)
        data = read_pts_file()
        data = move_center(data)
        data = normalize_data(data)
        for d in data:
            x = int(d[0]*200)+200
            y = int(d[1]*200)+200
            cv2.circle(img=cv_img, center = (x,y), radius =5, color =(0,255,0), thickness=-1)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        print(rgb_image.shape)
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)
    
if __name__=="__main__":
    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
from glob import glob
import numpy as np
import cv2
from sklearn.svm import SVC
import random
from math import cos, sin

class PatchModel:
    def __init__(self, file_img='indoor_001.png', file_pts='indoor_001.pts'):
        self.load_img(file_img)
        self.load_img_gray(file_img)
        self.load_pts(file_pts)
        # self._test_train_patch()
        self.weights = []
        self.__test_train_patch()

    def load_img(self, filename='indoor_001.png'):
        self.img = cv2.imread(filename)
        return self.img
    def load_pts(self, filename='indoor_001.pts'):
        """
        Read .pts file and return N*2-D data of facial landmark.

        Parameters
        -------------
        filename : string
        filename of .pts file.

        Returns
        -------------
        N*2-D numpy array of facial landmark
        """

        # parameters for facial landmark format
        pts_idx_start = 3
        pts_num = 68

        data =[]
        with open(filename) as f:
            lines = f.readlines()
            lines = lines[pts_idx_start:-1]
            for l in lines:
                x=float(l.replace('\n','').split(' ')[0])
                y=float(l.replace('\n','').split(' ')[1])
                data.append([x,y])
        data = np.array(data)

        assert data.shape[0] == pts_num
        self.pts=data
    def load_img_gray(self, filename='indoor_001.png'):
        self.img_gray = cv2.imread(filename)
        self.img_gray = cv2.cvtColor(self.img_gray, cv2.COLOR_BGR2GRAY)
        return self.img_gray
    def extract_patch(self, window_size=16):
        patches = []
        for pts_idx in range(0, self.pts.shape[0]):
            x = int(self.pts[pts_idx,0])
            y = int(self.pts[pts_idx,1])
            d = int(window_size/2)
            ret = self.img[y-d:y+d, x-d:x+d]
            patches.append(ret)
        patches = np.array(patches)
        patches.reshape(68,2*d,2*d,3)
        return patches
    def extract_patch_gray(self, window_size=16):
        patches = []
        for pts_idx in range(0, self.pts.shape[0]):
            x = int(self.pts[pts_idx,0])
            y = int(self.pts[pts_idx,1])
            d = int(window_size/2)
            ret = self.img_gray[y-d:y+d, x-d:x+d]
            patches.append(ret)
        patches = np.array(patches)

        patches.reshape(68,2*d,2*d)
        print(patches.shape)
        return patches 
    def _test_train_patch(self, idx=0):
        train_y = np.zeros(self.pts.shape[0])
        train_y[idx] = 1
        train_X = self.extract_patch_gray().reshape(68,16*16)
        svm = SVC(kernel='linear')
        svm.fit(train_X,train_y)
        self.weights = svm.coef_.reshape(16,16)
        self._weight_image = np.array(self.weights*10000+125).astype(np.uint8)
        print(type(self._weight_image))
        print(np.array(self.weights*10000+125).astype(np.uint8))
        print(svm.predict(train_X))
    def __test_train_patch(self, idx=0):
        files = glob('image/*.png')
        train_X = []
        train_y = []
        filecount=1
        num_files = 1
        actual_files=0
        for f in files[:num_files]:
            self.load_img(f)
            print('landmark_data/indoor_'+str(filecount).zfill(3)+'.pts')
            self.load_pts('landmark_data/indoor_'+str(filecount).zfill(3)+'.pts')
            filecount = filecount+1
            try:
                patches=self.extract_patch_gray()
                train_X.append(patches[idx])
                train_X.append(patches[random.randrange(1,68)])
                train_X.append(patches[random.randrange(1,68)])
                train_X.append(patches[random.randrange(1,68)])
                y = np.zeros(4)
                y[0]=1
                train_y.append(y)
                actual_files = actual_files+1
            except:
                continue
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        print(train_X.shape,train_y.shape)
        train_X = train_X.reshape(actual_files*4,16*16)
        train_y = train_y.reshape(actual_files*4)
        svm = SVC(C=1e-6,kernel='linear')
        svm.fit(train_X,train_y)
        self.weights = svm.coef_.reshape(16,16)
        self._weight_image = np.array(self.weights*1000000+125).astype(np.uint8)
        print(type(self._weight_image))
        print(np.array(self.weights*1000000+125).astype(np.uint8))
        print(svm.predict(train_X).reshape(actual_files,4))

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result
    def shift_pts(pts, vec):
        return pts + vec
    def rotate_pts(pts, angle):
        ret = []
        for p in pts:
            rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
            _p = np.dot(rot, p)
            ret.append(_p)
        return np.array(ret)
    def generate_cropped_data(self):
        files =['indoor_001.png'] #TODO: glob
        for f in files:
            img = self.load_img(f)
            pts = self.load_pts(f.split('.')[0]+'.pts')
            angle = 10 #TODO: angle here.
            rotated_img = self.rotate_image(img, angle)
            vec = pts[34]
            x = vec[0]
            y = vec[1]
            shifted_pts = self.shift_pts(pts, vec)
            rotate_pts = self.rotate_pts(shifted_pts, angle) + vec
            size = 100/2 #TODO: calculate from points!
            cropped_img = [y-size:y+size, x-size:x+size]
            rotate_pts = self.shift_pts(rotate_pts, -vec + size)

            #TODO: scale image + pts
            # export pts + image
            #rotate image by some degree
            
            


if __name__ == '__main__':
    patch_model = PatchModel()
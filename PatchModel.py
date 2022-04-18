import numpy as np
import cv2
from requests import patch


class PatchModel:
    def __init__(self, file_img='indoor_001.png', file_pts='indoor_001.pts'):
        self.load_img(file_img)
        self.load_pts(file_pts)
        self.weights = []

    def load_img(self, filename='indoor_001.png'):
        self.img = cv2.imread(filename)
        pass
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
    
    def extract_patch(self, window_size=9):
        patches = []
        for pts_idx in range(0, 68):
            x = int(self.pts[pts_idx,0])
            y = int(self.pts[pts_idx,1])
            d = int(window_size/2)
            ret = self.img[y-d:y+d, x-d:x+d]
            patches.append(ret)
        patches = np.array(patches)
        patches.reshape(68,8,8,3)
        return patches
        


if __name__ == '__main__':
    patch_model = PatchModel()
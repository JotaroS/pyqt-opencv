import numpy as np
from requests import get
import glob

def get_mean_face(dirname = 'landmark_data'):
    """
    get mean face from all .pts files in 'dirname'

    Parameters
    ----------
    dirname: name of directory of .pts files

    Returns
    ----------
    normalized mean face from all data
    """
    files = glob.glob(dirname+'/*.pts')
    mean = np.zeros(shape=(68,2))
    for file in files:
        data = read_pts_file(file)
        data = move_center(data)
        data = normalize_face(data)
        mean += data
    mean = mean / len(files)
    return mean

def normalize_face(data):
    """
    Normalize facial landmark data in the value of [-1, 1]

    Parameters
    ----------
    data: 2d matrix of facial landmark data

    Returns
    -----------
    data: 2d matrix of _normalized_ facial landmark data
    """
    w = data[:, 0].max() - data[:, 0].min()
    h = data[:, 1].max() - data[:, 1].min()
    data = data/np.array([w, h])
    return data

def move_center(data):
    """
    Shift the facial landmark to origin, based on 34-th facial landmark == center of nose tip

    ParamParameters
    ----------
    data: 2d matrix of facial landmark data

    Returns
    -----------
    data: 2d matrix of _shifted_ facial landmark dataetrs
    """

    assert data.shape==(68, 2)
    return data - data[34]

def read_pts_file(filename ='indoor_001.pts'):
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
    return data

            


if __name__ == '__main__':
    data = get_mean_face()
    print(data)
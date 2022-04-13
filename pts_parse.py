import numpy as np

def normalize_data(data):
    w = data[:, 0].max() - data[:, 0].min()
    h = data[:, 1].max() - data[:, 1].min()
    data = data/np.array([w, h])
    return data

def move_center(data):
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
    data=read_pts_file()
    data=move_center(data)
    data = normalize_data(data)
    print(data)
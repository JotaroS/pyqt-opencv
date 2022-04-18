import numpy as np
import glob
from sklearn.decomposition import PCA

class FaceModel:
    def __init__(self, filedir = 'landmark_data', n_components = 3):
        self.pc = self.get_principal_components(n_components)
        self.mu = self.get_mean_face()
        self.n_pc = n_components
        pass
    def get_all_face(self, dirname = 'landmark_data'):
        """
        get mean face from all .pts files in 'dirname'

        Parameters
        ----------
        dirname: name of directory of .pts files

        Returns
        ----------
        normalized mean face from all data
        """
        all_data = []
        files = glob.glob(dirname+'/*.pts')
        for file in files:
            data = self.read_pts_file(file)
            data = self.move_center(data)
            data = self.normalize_face(data)
            all_data.append(data)
        return np.array(all_data)

    def get_mean_face(self, dirname = 'landmark_data'):
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
            data = self.read_pts_file(file)
            data = self.move_center(data)
            data = self.normalize_face(data)
            mean += data
        mean = mean / len(files)
        return mean

    def normalize_face(self, data):
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

    def move_center(self, data):
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

    def read_pts_file(self, filename ='indoor_001.pts'):
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


    def get_principal_components(self, n_components = 3):
        """
        Gets principal components for each faces.

        Parameters:
        -------------
        n_components : number of principal components to obtain

        Return:
        -------------
        principal components: in 3 dimension vector data (n_components x 68 x 2)
        """
        data = self.get_all_face()
        data=data.reshape(data.shape[0], 68*2)
        pca = PCA(n_components = n_components)
        pca.fit(data)
        self.pc_eigen = pca.explained_variance_
        return pca.components_.reshape(n_components, 68, 2)

    def get_eigenvalues_per_face(self, face):
        """
        Gets eigenvalues for given facial data from constructed facial Point Distribution Model.
        (formally, calculates b = P.T(x-mu) for P as principal components and mu as mean face)

        Parameters:
        ------------
        face: facial data
        principal_components: principal components from facial data

        Return:
        ------------
        eigenvalues: eigenvalues for corresponding Principal component vectors.
        """

        n_components = self.pc.shape[0]
        # reshape p-components into n*136 positive-semi-definite matrix
        P = self.pc.reshape(n_components, 68*2)
        mu = self.mu.reshape(68*2)
        face = face.reshape(68*2)
        b = np.dot(P, face-mu)
        return b
    def get_deformation_factor(self, face):
        #TODO: calculate (sq-magnitude-eivenvector)/(eigenvalue). Does this function has to be here actually? (because it's not a part of facial PDM.)
        pass



if __name__ == '__main__':
    face_model = FaceModel()
    data = face_model.read_pts_file()
    data = face_model.move_center(data)
    data = face_model.normalize_face(data)
    print(face_model.get_eigenvalues_per_face(data))

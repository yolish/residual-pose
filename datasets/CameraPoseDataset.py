from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
from os.path import join
import numpy as np
import joblib


class CameraPoseDataset(Dataset):
    """
        A class representing a dataset of images and their poses
    """

    def __init__(self, dataset_path, labels_file, data_transform=None, position_num_classes=None,
                 orientation_num_classes=None, kmeans_position_file=None, kmeans_orientation_file=None):
        """
        :param dataset_path: (str) the path to the dataset
        :param labels_file: (str) a file with images and their path labels
        :param data_transform: (Transform object) a torchvision transform object
        :return: an instance of the class
        """
        super(CameraPoseDataset, self).__init__()
        self.img_paths, self.poses = read_labels_file(labels_file, dataset_path)
        self.dataset_size = self.poses.shape[0]
        self.predict_with_redisuals = False
        if position_num_classes is not None:
            self.predict_with_redisuals = True
            assert orientation_num_classes is not None
            if kmeans_position_file is None:
                from sklearn.cluster import KMeans
                random_state = 170

                # Incorrect number of clusters
                kmeans_position = KMeans(n_clusters=position_num_classes, random_state=random_state).fit(np.array(self.poses[:, :3]))
                kmeans_orientation = KMeans(n_clusters=orientation_num_classes, random_state=random_state).fit(self.poses[:, 3:])

                filename = labels_file + '_position_{}_classes.sav'.format(position_num_classes)
                joblib.dump(kmeans_position, filename)

                filename = labels_file + '_orientation_{}_classes.sav'.format(orientation_num_classes)
                joblib.dump(kmeans_orientation, filename)
            else:
                kmeans_position = joblib.load(kmeans_position_file)
                kmeans_orientation = joblib.load(kmeans_orientation_file)

            self.position_centroids = kmeans_position.cluster_centers_
            self.orientation_centroids = kmeans_orientation.cluster_centers_
            self.position_cls = kmeans_position.predict(self.poses[:, :3]).astype(np.int)
            self.orientation_cls = kmeans_orientation.predict(self.poses[:, 3:]).astype(np.int)

        self.transform = data_transform

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        img = imread(self.img_paths[idx])
        pose = self.poses[idx]
        if self.transform:
            img = self.transform(img)

        sample = {'img': img, 'pose': pose}
        if self.predict_with_redisuals:
            position_cls = self.position_cls[idx]
            sample['position_cls'] = position_cls
            sample['position_centroids'] = self.position_centroids

            orientation_cls = self.orientation_cls[idx]
            sample['orientation_cls'] = orientation_cls
            sample['orientation_centroids'] = self.orientation_centroids
        return sample


def read_labels_file(labels_file, dataset_path):
    df = pd.read_csv(labels_file)
    imgs_paths = [join(dataset_path, path) for path in df['img_path'].values]
    n = df.shape[0]
    poses = np.zeros((n, 7))
    poses[:, 0] = df['t1'].values
    poses[:, 1] = df['t2'].values
    poses[:, 2] = df['t3'].values
    poses[:, 3] = df['q1'].values
    poses[:, 4] = df['q2'].values
    poses[:, 5] = df['q3'].values
    poses[:, 6] = df['q4'].values
    return imgs_paths, poses
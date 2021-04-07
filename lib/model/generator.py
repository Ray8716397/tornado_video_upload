import numpy as np
import os
import pandas as pd
from keras.utils import Sequence
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatruesSequence(Sequence):
    """
    Thread-safe image generator with imgaug support

    For more information of imgaug see: https://github.com/aleju/imgaug
    """

    def __init__(self, features_name, features_subdir, batch_size=16):
        """
        :param dataset_csv_file: str, path of dataset csv file
        :param batch_size: int
        """
        self.features_name = features_name
        self.features_subdir = features_subdir
        self.features_dir = os.path.join('data/features/engineered', self.features_name, self.features_subdir)
        # self.x_path = sorted(os.listdir(self.features_dir))
        self.x_path = [os.path.join('data/features/engineered', self.features_name, self.features_subdir)]
        self.batch_size = batch_size
        self.steps = int(np.ceil(len(self.x_path) / float(self.batch_size)))

    def __bool__(self):
        return True

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        batch_x_path = self.x_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.asarray([self.load_openface_file(x_path) for x_path in batch_x_path])
        return batch_x

    def load_openface_file(self, openface_file):
        openface_path = os.path.join(self.features_dir, openface_file)
        features_array = pd.read_csv(openface_file).values
        stdsc = StandardScaler()
        features_array = stdsc.fit_transform(features_array)
        return features_array
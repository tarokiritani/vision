from __future__ import print_function
import os
import shutil
import tarfile

import scipy.io as sio

from .folder import ImageFolder, default_loader
from .utils import download_url, makedir_exist_ok

class StanfordCars(ImageFolder):

    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html/>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        class_names (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
    """

    urls = ["http://imagenet.stanford.edu/internal/car196/cars_train.tgz",
            "http://imagenet.stanford.edu/internal/car196/cars_test.tgz",
            "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            "http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat"]


    def __init__(self, root, split='train', transform=None, target_transform=None, loader=default_loader, **kwargs):
        self.root = root
        self.download()
        self.process()
        super(StanfordCars, self).__init__(os.path.join(self.processed_folder, split), transform=transform, target_transform=target_transform,
                loader=loader, **kwargs)
        
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def download(self):
        """Download the Stanford Cars data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)

    def _extract(self):
        print('extracting...')
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            if url[-4:] == '.tgz':
                tar = tarfile.open(file_path)
                tar.extractall(path=self.raw_folder)
                tar.close()
    
    def _check_exists(self):
        existence = []
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            existence.append(os.path.exists(file_path))
        return all(existence)

    def _check_processed(self):
        pass

    def process(self):
        """Parse the annotation files and copy image files in appropriate folders."""
        if self._check_processed():
            return
        self._extract()
        train_annotations = sio.loadmat(os.path.join(self.raw_folder, 'devkit/cars_train_annos.mat'))
        test_annotations = sio.loadmat(os.path.join(self.raw_folder, 'cars_test_annos_withlabels.mat'))
        mat_meta = sio.loadmat(os.path.join(self.raw_folder, 'devkit/cars_meta.mat'))
        self.class_names = mat_meta['class_names'][0]

        for test_annotation in test_annotations['annotations'][0]:
            class_id = test_annotation[-2][0][0]
            img_folder = os.path.join(self.processed_folder, 'test', self.class_names[class_id-1][0])
            makedir_exist_ok(img_folder)
            shutil.copy(os.path.join(self.raw_folder, 'cars_test', test_annotation[-1][0]), os.path.join(img_folder, test_annotation[-1][0]))

        
        for train_annotation in train_annotations['annotations'][0]:
            class_id = train_annotation[-2][0][0]
            img_folder = os.path.join(self.processed_folder, 'train', self.class_names[class_id-1][0])
            makedir_exist_ok(img_folder)
            shutil.copy(os.path.join(self.raw_folder, 'cars_train', train_annotation[-1][0]), os.path.join(img_folder, train_annotation[-1][0]))
               
        # process and save as torch files
        print('Done!')

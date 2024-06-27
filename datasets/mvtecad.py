import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from datasets.base_dataset import BaseADDataset
from datasets.cutmix import CutMix


class MVTecAD(BaseADDataset):

    def __init__(self, args, train=True):
        # Initialize the MVTecAD dataset.
        # Args:
        #     args: Command line arguments or configuration object.
        #     train (bool): Indicates whether the dataset is for training or testing.

        super().__init__()
        self.args = args
        self.train = train
        self.classname = args.classname
        self.know_class = args.know_class
        self.pollution_rate = args.cont_rate
        
        # Determine the test threshold
        if args.test_threshold == 0 and args.test_rate == 0:
            self.test_threshold = args.nAnomaly
        else:
            self.test_threshold = args.test_threshold

        self.root = os.path.join(args.dataset_root, self.classname)
        
        # Set transformations based on training or testing mode
        self.transform = self._get_transform()
        self.transform_pseudo = self._get_transform_pseudo()

        # Load normal training data
        normal_data = self._load_data('train')

        # Calculate the number of polluted samples
        self.nPollution = int(len(normal_data) / (1 - self.pollution_rate) * self.pollution_rate)
        
        # Adjust test threshold if necessary
        if self.test_threshold == 0 and args.test_rate > 0:
            self.test_threshold = int(len(normal_data) / (1 - args.test_rate) * args.test_rate) + args.nAnomaly

        self.ood_data = self._get_ood_data()

        # Load normal testing data if in testing mode
        if not self.train:
            normal_data = self._load_data('test')

        # Split the outlier data and combine it with normal data
        outlier_data, pollution_data = self._split_outlier()
        normal_data += pollution_data

        # Create labels for the data
        normal_labels = np.zeros(len(normal_data)).tolist()
        outlier_labels = np.ones(len(outlier_data)).tolist()

        # Combine data and labels
        self.images = normal_data + outlier_data
        self.labels = np.array(normal_labels + outlier_labels)
        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()

    def _load_data(self, split):
        # Load normal data from the specified split (train/test).
        # Args:
        #      split (str): The data split to load ('train' or 'test').
        # Returns:
        #      list: A list of file paths for normal data.

        data = []
        for file in os.listdir(os.path.join(self.root, split, 'good')):
            if file.lower().endswith(('png', 'jpg', 'npy')):
                data.append(f"{split}/good/{file}")
        return data

    def _get_ood_data(self):
        # Load out-of-distribution (OOD) data from other classes.
        # Returns:
        #     list: A list of file paths for OOD data.

        if self.args.outlier_root is None:
            return None
        ood_data = []
        for cls in os.listdir(self.args.outlier_root):
            if cls == self.classname:
                continue
            cls_path = os.path.join(self.args.outlier_root, cls, 'train', 'good')
            ood_data += [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.lower().endswith(('png', 'jpg', 'npy'))]
        return ood_data

    def _split_outlier(self):
        # Split the outlier data into known and unknown outliers.
        # Returns:
        #     tuple: Lists of known and unknown outlier data.

        outlier_data = []
        know_class_data = []
        outlier_classes = os.listdir(os.path.join(self.root, 'test'))
        
        for cls in outlier_classes:
            if cls == 'good':
                continue
            cls_files = [f"test/{cls}/{file}" for file in os.listdir(os.path.join(self.root, 'test', cls)) if file.lower().endswith(('png', 'jpg', 'npy'))]
            if cls == self.know_class:
                know_class_data += cls_files
            else:
                outlier_data += cls_files
        
        random_state = np.random.RandomState(self.args.ramdn_seed)
        random_state.shuffle(know_class_data)
        
        know_outlier = know_class_data[:self.args.nAnomaly]
        unknown_outlier = outlier_data
        
        if self.train:
            return know_outlier, []
        else:
            return unknown_outlier, []

    def _get_transform(self):
        # Get the data transformation for training or testing.
        # Returns:
        #     transforms.Compose: The composed transformations.

        if self.train:
            return transforms.Compose([
                transforms.Resize((self.args.img_size, self.args.img_size)),
                transforms.RandomRotation(180),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.args.img_size, self.args.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def _get_transform_pseudo(self):
        # Get the data transformation for pseudo-labeled data.
        # Returns:
        #     transforms.Compose: The composed transformations.

        return transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            CutMix(),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_image(self, path):        
        if path.lower().endswith('npy'):
            img = np.load(path).astype(np.uint8)
            img = img[:, :, :3]
            return Image.fromarray(img)
        return Image.open(path).convert('RGB')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Get a sample from the dataset at the specified index.
        # Args:
        #     index (int): The index of the sample.
        # Returns:
        #     dict: A dictionary containing the image and label.

        if self.train and index in self.normal_idx and random.randint(0, 1) == 0:
            if self.ood_data is None:
                index = random.choice(self.normal_idx)
                image = self.load_image(os.path.join(self.root, self.images[index]))
                transform = self.transform_pseudo
            else:
                image = self.load_image(random.choice(self.ood_data))
                transform = self.transform
            label = 2
        else:
            image = self.load_image(os.path.join(self.root, self.images[index]))
            transform = self.transform
            label = self.labels[index]
        return {'image': transform(image), 'label': label}

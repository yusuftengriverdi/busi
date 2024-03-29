from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
import cv2
import glob
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import os, sys
from .submodules.bbox import compute_bbox

parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from config import parse_arguments


# Base transform. 
# transpose = T.Lambda(lambda x: torch.permute(x, dims=(2, 0, 1)))

transform = T.Compose([T.ToTensor()])

def get_key_from_value(dict, value):
    """
    Get the key from a value in the label_mapping dictionary.
    Args:
        value: The value for which to find the key.
    Returns:
        The key corresponding to the given value.
    """
    for key, val in dict.items():
        try: 
            if val == value:
                return key
        except ValueError as e:
            if np.argmax(val) == np.argmax(value):
                return key
    raise ValueError(f"No key found for value: {value}")

    

def calculate_mean_std(cache_file="mean_std_cache.pth"):
    if os.path.exists(cache_file):
        # Load cached mean and std from file
        mean_std_dict = torch.load(cache_file)
        mean = mean_std_dict["mean"]
        std = mean_std_dict["std"]
    else:
        # Calculate mean and std
        means = []
        stds = []
        for item in tqdm(desc="Calculating mean and std"):
            img = item["image"]
            means.append(torch.mean(img))
            stds.append(torch.std(img))

        mean = torch.mean(torch.tensor(means))
        std = torch.mean(torch.tensor(stds))

        # Save mean and std to cache file
        mean_std_dict = {"mean": mean, "std": std}
        torch.save(mean_std_dict, cache_file)

    return mean, std


class CustomDataset(Dataset):
    """
    Custom PyTorch Dataset for reading image data with different label representations.
    Args:
        root_path (str): Root path of the dataset.
        mode (str): Mode for label representation ('0', '1', '2').
        include_normal (bool): Include normal cases in the dataset.
        image_size (tuple): Tuple containing width and height for image resizing.
    """
    def __init__(self, root_path, mode, include_normal, image_size, 
                 split_ratio=None, 
                 seed=None, 
                 transform=transform, 
                 mask_transform=transform,
                 log_dir= None, 
                 return_bbox = False,
                 ):
        
        self.label_mapping = {}
        self.weights = []
        self.root_path = root_path
        self.mode = mode
        self.include_normal = include_normal
        self.image_size = image_size
        self.data = self._read_data()
        self.split_ratio = split_ratio
        self.seed = seed
        # self.preprocessing = preprocessing
        self.transform = transform
        self.mask_transform = mask_transform
        # if split_ratio:
        #     self.split_data()
        self.dir = log_dir
        self.return_bbox = return_bbox

    def _read_data(self):
        """
        Reads image data from the dataset directory based on the specified mode.
        Returns:
            list: List of dictionaries containing image, label, and filename.
        """
        data = []

        if self.mode not in [0, 1, 2]:
            raise ValueError("Invalid mode. Mode must be one of: '0', '1', '2'.")

        if self.mode == 0:
            label_mapping = {
                'benign': 0,
                'malignant': 1,
                'normal': 2
            }
        elif self.mode == 1:
            label_mapping = {
                'benign': [1, 0, 0],
                'malignant': [0, 1, 0],
                'normal': [0, 0, 1]
            }


        # Reverse the mapping using tuples
        self.label_mapping = label_mapping

        # Read data for benign and malignant cases
        label_counter = {
                'benign': 0,
                'malignant': 0,
                'normal': 0,
                'sum': 0
            }
        
        for label in ['benign', 'malignant']:
            path_pattern = os.path.join(self.root_path, f'{label}/*.png')
            file_paths = glob.glob(path_pattern)
            for file_path in tqdm(file_paths, desc=f'Reading {label} cases'):
                if not '_mask' in file_path:
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, self.image_size)
                    label_value = label_mapping[label]

                    label_counter[label] += 1
                    label_counter['sum'] += 1

                    filename = os.path.basename(file_path).split('.')[0]

                    mask_path = file_path.replace('.png', '_mask.png')
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, self.image_size)
                        # Beta: Binarize again.
                        mask[mask > 256/2] = 255
                        mask[mask <= 256/2] = 0

                    else:
                        mask = None

                    data.append({'image': img, 
                                 'mask': mask, 
                                 'label': label_value, 
                                 'filename': filename})
                else:
                    continue

        # Read data for normal cases if include_normal is True
        if self.include_normal:
            path_pattern = os.path.join(self.root_path, 'normal/*.png')
            file_paths = glob.glob(path_pattern)
            for file_path in tqdm(file_paths, desc='Reading normal cases'):
                if not '_mask' in file_path:
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, self.image_size)
                    label_value = label_mapping['normal']
                    filename = os.path.basename(file_path).split('.')[0]

                    mask_path = file_path.replace('.png', '_mask.png')
                    if os.path.exists(mask_path):
                        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, self.image_size)
                    else:
                        mask = None

                    label_counter['normal'] += 1
                    label_counter['sum'] += 1
                    data.append({'image': img, 'mask': mask, 'label': label_value, 'filename': filename})
                else:
                    continue
        
        label_counter['benign'] /= label_counter['sum']
        label_counter['malignant'] /= label_counter['sum']
        label_counter['normal'] /= label_counter['sum']

        if self.include_normal:
            self.weights = list(label_counter.values())[:-1]
        else:
            self.weights = list(label_counter.values())[:-2]
        return data
    
    def set_transform(self, transform, mask_transform = None):

        self.transform = transform
        if mask_transform:
            self.mask_transform = mask_transform

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset at the given index.
        Args:
            idx (int): Index of the sample.
            dataset_type (str): Type of dataset ('train', 'val', 'test') when split_ratio is provided.
        Returns:
            dict: Dictionary containing image, mask (if available), label, and filename.
        """

        item = self.data[idx]

        image = item['image']
        mask = item.get('mask', None)
        label = item['label']
        filename = item['filename'] if 'filename' in item else None

        if self.transform:
            image = self.transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        if self.return_bbox:
            bbox = compute_bbox(mask[0])
            # print(bbox)
            # import matplotlib.pyplot as plt 
            # plt.imshow(mask[0])
            # plt.plot([bbox[0], bbox[2], bbox[2], bbox[0], bbox[0]],
            #         [bbox[1], bbox[1], bbox[3], bbox[3], bbox[1]],
            #         color='red', linewidth=2)
            # plt.show()
            return {'image': image, 'mask': mask, 'label': label, 'filename': filename, 'bbox': bbox}

        return {'image': image, 'mask': mask, 'label': label, 'filename': filename}
    

class SplitDataset(CustomDataset):
    def __init__(self, dataset, data = None, transform = None, mask_transform = None):
        self.transform = dataset.transform if not transform else transform
        self.mask_transform = dataset.mask_transform if not mask_transform else mask_transform
        self.return_bbox = dataset.return_bbox
        self.data = data

    def __getitem__(self, idx):
        # Override __getitem__ to apply the transformation
        item = super(SplitDataset, self).__getitem__(idx)
        
        return item

def get_data_loaders(args):

    split_ratio = None
    if args.SPLIT_RATIO:
        split_ratio = {'train': float(args.SPLIT_RATIO.split(':')[0]),
                       'val': float(args.SPLIT_RATIO.split(':')[1]),
                       'test': float(args.SPLIT_RATIO.split(':')[2])}

    
    # Create the dataset
    dataset = CustomDataset(
        root_path=args.ROOT,
        mode=args.MODE,
        include_normal=args.INCLUDE_NORMAL,
        image_size=(args.WIDTH, args.HEIGHT),
        seed=args.SEED,
        log_dir=args.DATE,
        return_bbox=args.return_bbox
    )

    if args.PRETRAINED:
        # Calculate or load mean and std
        mean, std = calculate_mean_std()

        normalize = T.Normalize(mean=mean, std=std)

        transform = T.Compose([T.ToPILImage(),T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
        
        # Do not normalize mask!
        mask_transform = T.Compose([T.ToPILImage(),T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        # Apply the transformation to the existing dataset
        dataset.set_transform(transform, mask_transform=mask_transform)
    
    # Splits the data into training, validation, and test sets based on the specified split ratio.    
    if args.SEED:
        np.random.seed(args.SEED)

    images = []
    masks = []
    labels = []
    filenames = []

    for item in dataset.data:
        images.append(item['image'])
        labels.append(item['label'])
        masks.append(item['mask'])
        filenames.append(item['filename'])

    images, labels, masks, filenames = np.array(images), np.array(labels), np.array(masks), np.array(filenames)

    if split_ratio['test'] != 0.0:
        train_data, test_data, train_masks, test_masks, train_labels, test_labels, train_filenames, test_filenames = train_test_split(
            images, masks, labels, filenames, test_size=split_ratio['test'], random_state=args.SEED, stratify=labels
        )

        train_data, val_data, train_masks, val_masks, train_labels, val_labels, train_filenames, val_filenames = train_test_split(
            train_data, train_masks, train_labels, train_filenames, test_size=split_ratio['val'],
            random_state=args.SEED, stratify=train_labels
        )

    else: 
        train_data, val_data, train_masks, val_masks, train_labels, val_labels, train_filenames, val_filenames = train_test_split(
            images, masks, labels, filenames, test_size=split_ratio['val'],
            random_state=args.SEED, stratify=labels
        )


    train_data = [{'image': img, 'mask': m, 'label': label, 'filename': filename}
                        for img, m, label, filename in zip(train_data, train_masks, train_labels, train_filenames)]
    val_data = [{'image': img, 'mask': m, 'label': label, 'filename': filename}
                        for img, m, label, filename in zip(val_data, val_masks, val_labels, val_filenames)]
    
    if split_ratio['test'] != 0.0:
        test_data = [{'image': img, 'mask': m, 'label': label, 'filename': filename}
                            for img, m, label, filename in zip(test_data, test_masks, test_labels, test_filenames)]

    else: 
        test_data = []

    # Saves filenames, sets, image sets, and labels to a CSV file.
    data = {'filename': [], 'set': [], 'label': []}


    for set_type, set_data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        for item in set_data:
            data['filename'].append(item['filename'])
            data['set'].append(set_type)
            data['label'].append(get_key_from_value(dataset.label_mapping, item['label']))

    df = pd.DataFrame(data)

    df.to_csv(f'training_runs/{dataset.dir}/fnames.csv', index=False)

    trainDataset = SplitDataset(dataset=dataset, data=train_data)
    valDataset = SplitDataset(dataset=dataset, data=train_data)

    if split_ratio['test'] != 0.0:
        testDataset = SplitDataset(dataset=dataset, data=train_data)
    
    # Example usage:
    # Accessing an example item
    sample_item = trainDataset[0]
    image = sample_item['image']
    mask = sample_item['mask']
    label = sample_item['label']
    filename = sample_item['filename']

    if args.return_bbox:
        bbox = sample_item['bbox']
    else: 
        bbox = None
    # Create DataLoaders
    
    train_dataloader = DataLoader(trainDataset, batch_size=args.BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(valDataset, args.BATCH_SIZE, shuffle=False)
    
    if split_ratio['test'] != 0.0:
        test_dataloader = DataLoader(testDataset, args.BATCH_SIZE, shuffle=False)
    else:
        test_dataloader = None

    with open(args.LOG, mode='a') as log_file:
        log_file.write("A sample image info: \n")
        log_file.write(f'Image shape: {image.shape}, Mask shape: {mask.shape if mask is not None else None},  \n'
            f'Label: {label}, Filename: {filename}, Bounding-box: {bbox} \n')
        
        log_file.write("Sizes of split sets --> \n ")
        log_file.write(f"Train: {len(train_dataloader) * args.BATCH_SIZE} \n")
        log_file.write(f"Validation: {len(val_dataloader) * args.BATCH_SIZE} \n")
        # log_file.write(f"Test: {len(test_dataloader) * args.BATCH_SIZE} \n")
        log_file.write("Dataset is ready!  \n")


    return train_dataloader, val_dataloader, test_dataloader, dataset.weights

if __name__ == '__main__':
    pass
# python prepare.py --ROOT /data/ --MODE 1 --INCLUDE_NORMAL --WIDTH 456 --HEIGHT 700

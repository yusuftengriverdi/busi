from tqdm import tqdm
import numpy as np
from torchvision import transforms as T
from torch.utils.data import DataLoader, TensorDataset
import torch
import datetime
import os
import pickle

from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y, mask, filename, transform = None):
        self.X = X
        self.y = y
        self.mask = mask
        self.filename = filename
        self.transform = transform

    def set_transform(self, transform):

        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        image = self.X[idx]
        mask = self.mask[idx]
        label = self.y[idx]
        filename = self.filename[idx]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'mask': mask, 'label': label, 'filename': filename}
    

def save_cache(cache, cache_path):
    """
    Save the cache to a file.

    Args:
    - cache: Data to be saved.
    - cache_path: Path to the cache file.
    """
    with open(cache_path, 'wb') as file:
        pickle.dump(cache, file)

def load_cache(cache_path):
    """
    Load the cache from a file.

    Args:
    - cache_path: Path to the cache file.

    Returns:
    - cache: Loaded data.
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as file:
            return pickle.load(file)
    return None

def augment(args, loader, ratio, cache_path='tmp/aug.pkl'):
    """
    Augment the data based on the specified ratio and return the original and augmented data loaders.

    Args:
    - args: Command line arguments containing the augmentation parameters.
    - loader: DataLoader for the original data.
    - ratio: Array representing the distribution of samples for each class.
    - cache_path: Path to the cache file.

    Returns:
    - combined_loader: DataLoader for the original and augmented data.
    """
    
    cache = load_cache(cache_path)

    if cache is None:

        # Calculate reverse distribution.
        n = args.BATCH_SIZE * len(loader)  # Approximate number

        dist = np.array([r * n for r in ratio])  # Ordered distribution of classes according to their majority.

        dist = np.flip((dist / 20 + 1 ).astype(int))  # Reverse sort replaces the distribution.

        print(dist)
        # Get to-be-augmented in a list.
        items = []

        for item in loader:
            X = item['image']
            y = item['label']
            mask = item['mask']
            filename = item['filename']

            if X.shape[0] == 32:
                items.append({'image': X, 'label': y, 'mask': mask, 'filename': filename})

        Xs = torch.cat([item['image'] for item in items], dim=0)
        ys = torch.cat([item['label'] for item in items], dim=0)
        masks = torch.cat([item['mask'] for item in items], dim=0)
        filenames = sum([item['filename'] for item in items], [])  # Concatenate list of lists


        # Define transforms.
        transform = T.Compose(transforms=[
            T.RandomVerticalFlip(p=0.5),
            # T.RandomHorizontalFlip(p=0.5),
            # T.RandomPerspective(p=0.1, distortion_scale=0.2),
            T.GaussianBlur(kernel_size=(3, 3)),
            # T.RandomRotation((0, 180))
        ])

        X_aug = torch.tensor([])
        y_aug = torch.tensor([])
        mask_aug = torch.tensor([])
        filename_aug = []

        for c in range(3):
            if args.MODE == 1:
                indices = torch.where(torch.argmax(ys, dim=1) == c)
                X_class = Xs[indices[0], :, :, :]
            else:
                X_class = Xs[torch.where(ys == c)]

            print("the class ratio", X_class.shape, Xs.shape)
            X_transform = []
            mask_transform = []
            for _ in tqdm(range(dist[c])):
                X_transform += [transform(X_class)] # 306 * 98
                mask_transform +=  [masks[indices[0]]]
                filename_aug += [filenames[i] for i in indices[0]]
            print("new fname shape", len(filename_aug)) 

            X_transform, mask_transform = np.array(X_transform), np.array(mask_transform)

            s = X_transform.shape
            X_transform = torch.tensor(X_transform.reshape(s[0]* s[1], s[2], s[3], s[4]))
            print("new shape", X_transform.shape, X_transform[0].shape) 

            s = mask_transform.shape
            print(s)
            mask_transform = torch.tensor(mask_transform.reshape(s[0]* s[1], s[2], s[3]))

            X_aug = torch.cat([X_aug, X_transform], dim=0)
            print("new combined x shape", X_aug.shape, X_aug[0].shape) 
            
            y_aug = torch.cat([y_aug, torch.ones(size=(len(X_transform),))*c], dim=0)
            print("new combined y shape", y_aug.shape, y_aug[0].shape) 

            mask_aug = torch.cat([mask_aug, mask_transform], dim=0)
            print("new mask shape", mask_aug.shape, mask_transform.shape, mask_transform[0].shape) 

        # Return one-hot vectors if MODE is 1; otherwise, return class indices.
        if args.MODE == 1:
            y_aug = torch.stack([torch.eye(3)[int(_)] for _ in y_aug])

        # Combine augmented and original data
        X_combined = torch.cat([Xs, X_aug], dim=0)
        y_combined = torch.cat([ys, y_aug], dim=0)
        mask_combined = torch.cat([masks, mask_aug], dim=0)
        filename_combined = filenames + filename_aug
        # Create DataLoader for combined data
        print(X_combined.shape, y_combined.shape)
        combined_dataset = CustomDataset(X_combined, y_combined, mask_combined, filename_combined)

        if args.PRETRAINED:
            means = []
            stds = []
            for item in combined_dataset:
                img = item["image"]
                means.append(torch.mean(img))
                stds.append(torch.std(img))

            mean = torch.mean(torch.tensor(means))
            std = torch.mean(torch.tensor(stds))

            normalize = T.Normalize(mean=mean, std=std)

            transform = T.Compose([T.ToPILImage(),T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])
            # Apply the transformation to the existing dataset
            combined_dataset.set_transform(transform)

        combined_loader = DataLoader(combined_dataset, batch_size=args.BATCH_SIZE, shuffle=True)

        # Log augmented samples
        log_file_path = f"tmp/augmentation_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(log_file_path, 'a') as log_file:
            for c, count in enumerate(dist):
                log_file.write(f"Class {c}: Augmented {count} samples\n")

        # Save the updated cache
        save_cache(combined_loader, cache_path)

        return combined_loader

    else:
        return cache

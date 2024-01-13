import numpy as np
from patchify import patchify
from tqdm import tqdm 

## Patch before training for segmentation
def patch(item, patch_size = None):

    if not patch_size:
        patch_size = (75, 75)

    patches = []

    X = item['image']
    y = item['label']
    m = item['mask']
    fname = item['filename']

    print(X.shape)
    for idx in range(len(item)):
        X_patches = patchify(X[idx].detach().cpu().numpy(), (patch_size), step=patch_size[0] ).reshape((-1, patch_size[0], patch_size[1]))
        m_patches = patchify(m[idx].detach().cpu().numpy(), (patch_size), step=patch_size[0] ).reshape((-1, patch_size[0], patch_size[1]))

        for X_patch, m_patch in zip(X_patches, m_patches):
            # Check if m_patch contains values other than 0
            if np.any(m_patch != 0):
                patches.append({'image': X_patch, 'label': y, 'filename': fname, 'mask': m_patch})
            
    return patches


import matplotlib.pyplot as plt

def visualize_patches(patches, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))

    for i in range(rows):
        for j in range(cols):
            index = i * cols + j
            if index < len(patches):
                patch = patches[index]['image']  # Extract the image from the patch
                label = patches[index]['label']
                mask = patches[index]['mask']   # Extract the mask from the patch

                axes[i, j].imshow(patch, cmap='gray')
                axes[i, j].contour(mask, colors='red', levels=[0.5])  # Visualize the mask as contours
                axes[i, j].set_title(f'Label: {label}')

            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Example usage:
    # Assuming patches is a list of patches returned from the patch function
    from prepare import CustomDataset

    split_ratio = {'train': 0.7,
                    'val': 0.2,
                    'test': 0.1}
    # Create the dataset
    dataset = CustomDataset(
        root_path='data/',
        mode=1, 
        include_normal=False,
        image_size=(512, 512),
        split_ratio=split_ratio,
        seed=42,
        log_dir='',
        preprocessing=True)

    patches = patch(dataset.data[0])
    visualize_patches(patches, rows=2, cols=4)

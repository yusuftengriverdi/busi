import torch

def compute_bbox(mask):
    # Assuming mask is a binary mask with shape [H, W]

    # Convert the mask to indices
    indices = torch.nonzero(mask)

    if indices.size(0) == 0:
        # No non-zero elements in the mask, return an empty tensor
        return torch.empty(0, 4)

    # Compute the bounding box coordinates
    x_min = torch.min(indices[:, 1])
    y_min = torch.min(indices[:, 0])
    x_max = torch.max(indices[:, 1])
    y_max = torch.max(indices[:, 0])

    # Ensure 0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(mask.shape[1], x_max + 1)
    y_max = min(mask.shape[0], y_max + 1)

    # Create the bounding box tensor [x1, y1, x2, y2]
    bbox = torch.tensor([[x_min, y_min, x_max, y_max]], dtype=torch.float32)

    return bbox

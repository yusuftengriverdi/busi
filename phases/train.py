import torchvision.models as models
from torchvision import transforms as T
import torch
from torch import optim
from tqdm import tqdm 
from torchmetrics import Accuracy, AUROC, F1Score, ConfusionMatrix
import os, csv
import time

def calculate_metrics(y, yhat, mode, device, label_mode=1):
    """
    Calculate evaluation metrics given ground truth (y) and predicted values (yhat).
    Args:
        y (torch.Tensor): Ground truth labels.
        yhat (torch.Tensor): Predicted labels.
        mode (str): 'binary' or 'multiclass'.
        device (str): Device ('cpu' or 'cuda').
        label_mode (int): Label mode (1 for integer labels, 0 for one-hot encoded labels).
    Returns:
        dict: Dictionary containing computed metrics.
    """
    if mode == 'multiclass': num_classes = 3
    else:num_classes = 2
    eval_metrics = {
        'accuracy_score': Accuracy(task=mode, num_classes=num_classes).to(device),
        'roc_auc_score': AUROC(task=mode, num_classes=num_classes).to(device),
        'f1_score': F1Score(mode, num_classes=num_classes).to(device),
        # 'confusion': ConfusionMatrix(mode).to(device)
    }

    scores = {
        'accuracy_score': 0.0,
        'roc_auc_score': 0.0,
        'f1_score': 0.0,
        # 'confusion': torch.zeros((3, 3), dtype=float)
    }

    # Compute evaluation metrics
    with torch.no_grad():

        for metric_name, metric in eval_metrics.items():
            if metric_name == 'accuracy_score':
                y_max = torch.argmax(y, dim=1).to(device)
                metric_val = metric(yhat, y_max)
            elif metric_name == 'roc_auc_score':
                y_max = torch.argmax(y, dim=1).to(device)
                metric_val = metric(yhat, y_max)
            elif metric_name == 'f1_score':
                y_max = torch.argmax(y, dim=1).to(device)
                metric_val = metric(yhat, y_max)
            else: 
                raise NotImplementedError

            scores[metric_name] += metric_val.item()

        return scores



def train(args, train_loader, val_loader, weights):
    """
    Train the model using the specified arguments and data loaders.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        weights (list): List of class weights for CrossEntropyLoss.
    """

    # Directory to store training runs
    run_dir = os.path.join("training_runs", args.DATE)
    os.makedirs(run_dir, exist_ok=True)

    # CSV file to log metrics
    csv_file_path = os.path.join(run_dir, "metrics.csv")

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        headers = ['Epoch', 'Train Loss'] + [f'Train {metric}' for metric in ['Acc Score', 'ROCAUC Score', 'f1 Score']] + \
                  ['Val Loss'] + [f'Val {metric}' for metric in  ['Acc Score', 'ROCAUC Score', 'f1 Score']]+ ['Computational Time (m)']
        writer.writerow(headers)

    # Call model
    model = models.resnet18(pretrained=args.PRETRAINED)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, args.num_classes)

    device = args.TO

    if device != 'cpu':
        model = torch.nn.DataParallel(model.to(device))

    # Call optimizer.
    if args.OPT == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.LR,
                              weight_decay=args.MOMENTUM)
    else:
        raise NotImplementedError

    with open(args.LOG, mode='a') as log_file:
        log_file.write(f"Class weights are calculated as following and will be used in Loss function. {weights}  \n")

    if args.LOSS == 'CrossEntropy':
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights)).to(device)


    mode = 'binary' if args.num_classes == 2 else 'multiclass'

    average_loss = 0.0
    for ep in tqdm(range(args.EP), unit='epoch'):

        start = time.time()
        # Train.
        train_loss = 0
        train_metrics = {
        'accuracy_score': 0.0,
        'roc_auc_score': 0.0,
        'f1_score': 0.0,
        # 'confusion': torch.zeros((3, 3), dtype=float)
        }

        with open(args.LOG, mode='a') as log_file:
            log_file.write(f"Epoch {ep}, Avg. Loss {average_loss}  \n")
        for batch, item in enumerate(train_loader):

            X = item['image']
            y = item['label']

            if not args.PRETRAINED:
                X = X / 255.0
            optimizer.zero_grad()

            X = X.permute(0, -1, 1, 2)
            X = X.to(device)
            y = y.float().to(device)

            X = X.requires_grad_()
            yhat = model(X).to(device)

            loss = criterion(yhat, y)
            y = y.long()

            # Update model, gradient descent.
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Method 2: Using dictionary comprehension
            train_metrics = {k: v + calculate_metrics(y, yhat, mode, device)[k] for k, v in train_metrics.items()}

            if batch % 10 == 0:
                average_loss = train_loss / (batch + 1)

        train_loss /= len(train_loader)
        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}


        # Validate
        val_loss = 0
        val_metrics = {
        'accuracy_score': 0.0,
        'roc_auc_score': 0.0,
        'f1_score': 0.0,
        # 'confusion': torch.zeros((3, 3), dtype=float)
        }
        for batch, item in enumerate(val_loader):

            X = item['image']
            y = item['label']

            if not args.PRETRAINED:
                X = X / 255.0

            X = X.permute(0, -1, 1, 2)
            X = X.to(device)
            y = y.float().to(device)

            with torch.no_grad():
                X = X.requires_grad_()
                yhat = model(X)

                loss = criterion(yhat, y)
                
                y = y.long()

                val_loss += loss.item()

                val_metrics = {k: v + calculate_metrics(y, yhat, mode, device)[k] for k, v in val_metrics.items()}

        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
    
        computation_time = (time.time() - start) / 60.0
        # Append metrics to CSV
        row = [ep + 1, train_loss] + list(train_metrics.values()) + [val_loss] + list(val_metrics.values()) + [computation_time]
    
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    with open(args.LOG, mode='a') as log_file:
        log_file.write(f"Metrics logged in: {csv_file_path}  \n")

    return model

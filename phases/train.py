import torchvision.models as models
import torch
from torch import optim
from tqdm import tqdm 
from torchmetrics import Accuracy, AUROC, F1Score
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision
import os, csv
import time
from .losses.focal import FocalLoss
from .models.fpcn import FPCN
from .submodules import attention
from .submodules.backboned_unet import backboned_unet as unet 


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
        'macro_acc_score': MulticlassAccuracy(average='macro', num_classes=num_classes),
        'macro_f1_score': MulticlassF1Score(average='macro', num_classes=num_classes),
        'macro_precision_score': MulticlassPrecision(average='macro', num_classes=num_classes),
        'class_acc_score': MulticlassAccuracy(average=None, num_classes=num_classes),
        'class_f1_score': MulticlassF1Score(average=None, num_classes=num_classes)
        # 'confusion': ConfusionMatrix(mode, num_classes=num_classes).to(device)
    }

    scores = {
        'accuracy_score': 0.0,
        'roc_auc_score': 0.0,
        'f1_score': 0.0,
        'macro_acc_score': 0.0,
        'macro_f1_score': 0.0,
        'macro_precision_score': 0.0,
        'class_acc_score': torch.tensor([0.0 for c in range(num_classes)]),
        'class_f1_score': torch.tensor([0.0 for c in range(num_classes)])
        # 'confusion': torch.zeros((3, 3), dtype=float)
    }

    # Compute evaluation metrics
    with torch.no_grad():

        for metric_name, metric in eval_metrics.items():
            
            try:
                y_max = torch.argmax(y, dim=1).to(device)

                if 'macro' in metric_name or 'class' in metric_name:
                    metric.update(yhat, y_max)
                    metric_val = metric.compute()
                    if 'class' in metric_name:
                        # for i in range(num_classes):
                        #     print(metric_name, scores[metric_name], metric_val)
                        #     scores[metric_name][i] += metric_val[i]
                        scores[metric_name] += metric_val
                    else:
                        scores[metric_name] += metric_val.item()

                else:
                    metric_val = metric(yhat, y_max)
                    scores[metric_name] += metric_val.item()


            except Exception as e:
                print(str(e))
                raise NotImplementedError


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

    metric_titles = ['Acc Score', 'ROCAUC Score', 'f1 Score', 
                     'Macro Acc Score', 'Macro f1 Score',  'Macro Precision Score',
                     'Class Acc Scores', 'Class f1 Scores']

    # Directory to store training runs
    run_dir = os.path.join("training_runs", args.DATE)
    os.makedirs(run_dir, exist_ok=True)

    # CSV file to log metrics
    csv_file_path = os.path.join(run_dir, "metrics.csv")

    with open(csv_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        headers = ['Epoch', 'Train Loss'] + [f'Train {metric}' for metric in metric_titles] + \
                  ['Val Loss'] + [f'Val {metric}' for metric in  metric_titles]+ ['Computational Time (m)']
        writer.writerow(headers)

    # Call model
    if args.TASK == 'Classify':

        if args.MODEL == 'Resnet18':
            if not args.ATTENTION: 
                model = models.resnet18(pretrained=args.PRETRAINED)
                num_ftrs = model.fc.in_features
                model.fc = torch.nn.Linear(num_ftrs, args.num_classes)
            else:
                model = attention.resnet18attention(num_classes=args.num_classes, use_mask=True, scale_dot_product=True, pretrained=args.PRETRAINED)

        elif args.MODEL == 'FPCN':
            model =  FPCN(num_classes=args.num_classes, use_pretrained=True)

    elif args.TASK == 'Segment':
        
        if args.MODEL == 'Unet':
            model = unet.unet.Unet(backbone_name='resnet18', pretrained=args.PRETRAINED, classes=args.num_classes)

    device = args.TO

    weights = torch.tensor(weights).to(device)

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
        criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
    elif args.LOSS == 'Focal':
        criterion = FocalLoss(weight=weights, alpha=0.5).to(device)

    mode = 'binary' if args.num_classes == 2 else 'multiclass'

    average_loss = 0.0
    for ep in tqdm(range(args.EP), unit='epoch'):

        start = time.time()
        # Train.
        train_loss = 0

        if mode == 'multiclass':
            train_metrics = {
                'accuracy_score': 0.0,
                'roc_auc_score': 0.0,
                'f1_score': 0.0,
                'macro_acc_score': 0.0,
                'macro_f1_score': 0.0,
                'macro_precision_score': 0.0,
                'class_acc_score': torch.tensor([0.0 for c in range(args.num_classes)]),
                'class_f1_score': torch.tensor([0.0 for c in range(args.num_classes)])
            }
        else:
            train_metrics = {
                'accuracy_score': 0.0,
                'roc_auc_score': 0.0,
                'f1_score': 0.0,
            }

        with open(args.LOG, mode='a') as log_file:
            log_file.write(f"Epoch {ep}, Avg. Loss {average_loss}  \n")
        for batch, item in enumerate(train_loader):

            X = item['image']
            y = item['label']
            m = item['mask']

            X = X / 255.0
            optimizer.zero_grad()

            X = X.to(device)
            y = y.float().to(device)

            X = X.float().to(device).requires_grad_()

            if not args.USE_MASK:
                yhat = model(X).to(device)
            else: 
                yhat = model(X, mask=m).to(device)

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
        if mode == 'multiclass':
            val_metrics = {
                'accuracy_score': 0.0,
                'roc_auc_score': 0.0,
                'f1_score': 0.0,
                'macro_acc_score': 0.0,
                'macro_f1_score': 0.0,
                'macro_precision_score': 0.0,
                'class_acc_score': torch.tensor([0.0 for c in range(args.num_classes)]),
                'class_f1_score': torch.tensor([0.0 for c in range(args.num_classes)])
            }
        else:
            val_metrics = {
                'accuracy_score': 0.0,
                'roc_auc_score': 0.0,
                'f1_score': 0.0,
            }

        for batch, item in enumerate(val_loader):

            X = item['image']
            y = item['label']

            # if not args.PRETRAINED:
            X = X / 255.0

            # X = X.permute(0, -1, 1, 2)
            X = X.to(device)
            y = y.float().to(device)

            with torch.no_grad():
                X = X.float().to(device)
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

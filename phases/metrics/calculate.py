from torchmetrics import Accuracy, AUROC, F1Score, Dice
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassAUROC
import torch
# add to the system path
import sys
from ..losses.dice import DiceLoss

def calculate_clf_metrics(y, yhat, mode, device):
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
    
    eval_clf_metrics = {
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

        for metric_name, metric in eval_clf_metrics.items():
            
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


def calculate_seg_metrics(y, yhat, mode, device):
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
    
    eval_seg_metrics = {
        'accuracy_score': Accuracy(task=mode, num_classes=num_classes).to(device),
        'dice_score': DiceLoss(return_score=True),
        'roc_auc_score': AUROC(task=mode, num_classes=num_classes).to(device),
        # 'cohen_kappa_score': segmetrics.kapp_score_Value,
        # 'mse_log_error': segmetrics.mse,
        # 'nmi_score': segmetrics.nmi,
        # 'roc_auc_score': segmetrics.roc_auc,
    }

    scores = {
        'accuracy_score': 0.0,
        'dice_score': 0.0,
        'roc_auc_score': 0.0
        # 'cohen_kappa_score': 0.0,
        # 'mse_log_error': 0.0,
        # 'nmi_score': 0.0,
        # 'roc_auc_score': 0.0,
    }

    # Compute evaluation metrics
    with torch.no_grad():

        for metric_name, metric in eval_seg_metrics.items():
            
            try:
                # yhat = torch.argmax(yhat, dim=1).unsqueeze(dim=1)
                metric_val = metric(yhat, y.int())
                scores[metric_name] += metric_val

            except Exception as e:
                print(str(e))
                print(metric_name, metric)
                raise NotImplementedError

        return scores


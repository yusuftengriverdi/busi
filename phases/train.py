import torchvision.models as models
import torch
from torch import optim
from tqdm import tqdm 

import os, csv
import time
from .losses.focal import FocalLoss
from .models.fpcn import FPCN
from .submodules import attention
from .submodules.backboned_unet import backboned_unet as unet 
from .patch import patch
from .metrics import calculate

def train(args, train_loader, val_loader, weights):
    """
    Train the model using the specified arguments and data loaders.
    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        weights (list): List of class weights for CrossEntropyLoss.
    """

    if args.TASK == 'Classify':
        metric_titles = ['Acc Score', 'ROCAUC Score', 'f1 Score', 
                     'Macro Acc Score', 'Macro f1 Score',  'Macro Precision Score',
                     'Class Acc Scores', 'Class f1 Scores']
    elif args.TASK == 'Segment':
        metric_titles = ['Acc Score', 'Dice Score', 'ROCAUC Score',
                         ]
    elif args.TASK == 'Both':
        metric_titles = ['Acc Score', 'Dice Score', 'ROCAUC Score', 'f1 Score', 
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
        
        elif args.MODEL == 'EfficientNet':
            if not args.ATTENTION: 
                model = models.efficientnet_b0(pretrained=args.PRETRAINED)
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = torch.nn.Linear(num_ftrs, args.num_classes)
            else:
                model = attention.efficientb0attention(num_classes=args.num_classes, use_mask=True, scale_dot_product=True, pretrained=args.PRETRAINED)

        else:
            raise ValueError("Please select a valid model for Classification problem.")

    elif args.TASK == 'Segment':
        if args.MODEL == 'Unet':
            model = unet.unet.Unet(backbone_name=args.BACKBONE, pretrained=args.PRETRAINED, classes=args.num_classes)
        elif args.MODEL == 'DeepLabv3':
            model = models.segmentation.deeplabv3_resnet50(pretrained=args.PRETRAINED, aux_loss=False)
            model.classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
            model.aux_classifier[4] = torch.nn.Conv2d(256, 2, kernel_size=(1,1), stride=(1,1))
        else:
            raise ValueError("Please select a valid model for Segmentification problem.")

    elif args.TASK == 'Both':
        if args.MODEL == 'Retina':
            model = models.detection.retinanet_resnet50_fpn(pretrained=args.PRETRAINED, 
                                                            num_classes=args.num_classes,
                                                            pretrained_backbone=True)
        if args.MODEL == 'MaskRCNN':
            model = models.detection.maskrcnn_resnet50_fpn(pretrained=args.PRETRAINED, 
                                                            num_classes=args.num_classes)
        else:
            raise ValueError("Please select a valid model for Segmentification problem.")

    device = args.TO

    weights = torch.tensor(weights).to(device)

    if device != 'cpu':
        model = torch.nn.DataParallel(model.to(device))

    # Call optimizer.
    if args.OPT == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                              lr=args.LR,
                              weight_decay=args.MOMENTUM)
    elif args.OPT == 'Adam':
        optimizer = optim.Adam(model.parameters(),
                              lr=args.LR,
                              weight_decay=args.MOMENTUM)
    else:
        raise NotImplementedError
    
    if args.USE_SCHEDULER:
        from torch.optim.lr_scheduler import StepLR
        # Define the learning rate scheduler
        scheduler = StepLR(optimizer, step_size=args.LR_STEP_SIZE, gamma=args.LR_GAMMA)

    with open(args.LOG, mode='a') as log_file:
        log_file.write(f"Class weights are calculated as following and will be used in Loss function. {weights}  \n")

    if args.TASK == 'Classify':
        if args.LOSS == 'CrossEntropy':
            criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        elif args.LOSS == 'Focal':
            criterion = FocalLoss(weight=weights, alpha=0.5).to(device)
        else:
            raise ValueError("Please select a valid criterion for Classification problem.")

    elif args.TASK == 'Segment':
        if args.LOSS == 'Dice':
            raise NotImplementedError
        elif args.LOSS == 'MSE':
            criterion = torch.nn.MSELoss(reduction='sum')
        elif args.LOSS == 'CrossEntropy':
            criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)
        elif args.LOSS == 'DiceCE':
            raise NotImplementedError
        else:
            raise ValueError("Please select a valid criterion for Classification problem.")

    else:
        criterion = torch.nn.CrossEntropyLoss(weight=weights).to(device)

    mode = 'binary' if args.num_classes == 2 else 'multiclass'

    average_loss = 0.0
    # Add this variable to track the best val accuracy
    best_val_acc = 0.0

    
    for ep in tqdm(range(args.EP), unit='epoch'):

        start = time.time()
        # Train.
        train_loss = 0

        if args.TASK == 'Classify':
            if mode == 'multiclass':
                train_metrics = {
                    'accuracy_score': 0.0,
                    'roc_auc_score': 0.0,
                    'f1_score': 0.0,
                    'macro_acc_score': 0.0,
                    'macro_f1_score': 0.0,
                    'macro_precision_score': 0.0,
                    'class_acc_score': torch.tensor([0.0 for _ in range(args.num_classes)]),
                    'class_f1_score': torch.tensor([0.0 for _ in range(args.num_classes)])
                }
            else:
                train_metrics = {
                    'accuracy_score': 0.0,
                    'roc_auc_score': 0.0,
                    'f1_score': 0.0,
                }
        elif args.TASK == 'Segment':
            train_metrics =  {  
                'accuracy_score': 0.0,
                'dice_score': 0.0,
                'roc_auc_score': 0.0
        }

        with open(args.LOG, mode='a') as log_file:
            log_file.write(f"Epoch {ep}, Avg. Loss {average_loss}  \n")
        for batch, item in enumerate(train_loader):

            if args.USE_PATCH: 
                item = patch(item, patch_size=(args.PATCH_SQUARE_SIZE, args.PATCH_SQUARE_SIZE))
            
            
            X = item['image']
            y = item['label']
            m_ = item['mask'].to(device)

            if args.return_bbox:
                b = item['bbox']

            if args.TASK in ['Segment', 'Both']:
                # Create a tensor filled with zeros of shape (batch_size, num_classes, w, h)
                m = torch.zeros((m_.shape[0], args.num_classes, m_.shape[2], m_.shape[3]), device=device)

                # Iterate over classes and set the corresponding channel to 1 where m_ equals the class index
                for class_index in range(args.num_classes):
                    m[:, class_index, :, :] = (m_ == class_index).float().squeeze(1)


            X = X / 255.0
            optimizer.zero_grad()

            X = X.to(device)
            y = y.float().to(device)

            X = X.float().to(device).requires_grad_()

            if args.TASK != 'Both':
                if not args.USE_MASK:
                    if args.MODEL != 'DeepLabv3':
                        yhat = model(X).to(device)
                    else:
                        yhat = model(X)['out'].to(device)

                else: 
                    yhat = model(X, mask=m_).to(device)
            
            else:
                y = y.to(torch.int64)
                yhat = model(X, [{'boxes': b[i], 'labels': y[i], 'masks': m[i]} for i in range(args.BATCH_SIZE)])
            

            if args.TASK == 'Classify':
                loss = criterion(yhat, y)
                y = y.long()
                train_metrics = {k: v + calculate.calculate_clf_metrics(y, yhat, mode, device)[k] for k, v in train_metrics.items()}

            elif args.TASK == 'Segment':
                loss = criterion(yhat, m)
                train_metrics = {k: v + calculate.calculate_seg_metrics(m, yhat, mode, device)[k] for k, v in train_metrics.items()}
            else:
                raise NotImplementedError
            

            # Update model, gradient descent.
            loss.backward()
            optimizer.step()
            
            if args.USE_SCHEDULER:
                # Adjust learning rate
                scheduler.step()

            train_loss += loss.item()

            # Method 2: Using dictionary comprehension

            if batch % 10 == 0:
                average_loss = train_loss / (batch + 1)


        train_loss /= len(train_loader)

        train_metrics = {k: v / len(train_loader) for k, v in train_metrics.items()}


        # Validate
        val_loss = 0
        if args.TASK == 'Classify':
            if mode == 'multiclass':
                val_metrics = {
                    'accuracy_score': 0.0,
                    'roc_auc_score': 0.0,
                    'f1_score': 0.0,
                    'macro_acc_score': 0.0,
                    'macro_f1_score': 0.0,
                    'macro_precision_score': 0.0,
                    'class_acc_score': torch.tensor([0.0 for _ in range(args.num_classes)]),
                    'class_f1_score': torch.tensor([0.0 for _ in range(args.num_classes)])
                }
            else:
                val_metrics = {
                    'accuracy_score': 0.0,
                    'roc_auc_score': 0.0,
                    'f1_score': 0.0,
                }
        elif args.TASK == 'Segment':
                val_metrics = {        
                    'accuracy_score': 0.0,
                    'dice_score': 0.0,
                    'roc_auc_score': 0.0
        # 'cohen_kappa_score': 0.0,
        # 'mse_log_error': 0.0,
        # 'nmi_score': 0.0,
        # 'roc_auc_score': 0.0,
        }

        for batch, item in enumerate(val_loader):

            X = item['image']
            y = item['label']
            m_ = item['mask'].to(device)

            # Create a tensor filled with zeros of shape (batch_size, num_classes, w, h)
            m = torch.zeros((m_.shape[0], args.num_classes, m_.shape[2], m_.shape[3]), device=device)

            # Iterate over classes and set the corresponding channel to 1 where m_ equals the class index
            for class_index in range(args.num_classes):
                m[:, class_index, :, :] = (m_ == class_index).float().squeeze(1)

            # print("Mask properties: ", m.shape, torch.unique(m))
            # if not args.PRETRAINED:
            X = X / 255.0

            # X = X.permute(0, -1, 1, 2)
            X = X.to(device)
            y = y.float().to(device)

            with torch.no_grad():
                X = X.float().to(device)
                if args.MODEL != 'DeepLabv3':
                    yhat = model(X).to(device)
                else:
                    yhat = model(X)['out'].to(device)


                if args.TASK == 'Classify':
                    loss = criterion(yhat, y)
                    y = y.long()
                    val_metrics = {k: v + calculate.calculate_clf_metrics(y, yhat, mode, device)[k] for k, v in val_metrics.items()}
                elif args.TASK == 'Segment':
                    loss = criterion(yhat, m)
                    val_metrics = {k: v + calculate.calculate_seg_metrics(m, yhat, mode, device)[k] for k, v in val_metrics.items()}

                else:
                    raise NotImplementedError
                
                val_loss += loss.item()


        val_loss /= len(val_loader)
        val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}

        computation_time = (time.time() - start) / 60.0

        # Check if the current model has the best validation loss
        if val_metrics['accuracy_score'] > best_val_acc:
            best_val_acc = val_metrics['accuracy_score'] 
            # Save the model
            torch.save(model.state_dict(), f'{args.LOG_DIR}/best_val_acc_model.pth')

        # Append metrics to CSV
        row = [ep + 1, train_loss] + list(train_metrics.values()) + [val_loss] + list(val_metrics.values()) + [computation_time]
    
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(row)

    with open(args.LOG, mode='a') as log_file:
        log_file.write(f"Metrics logged in: {csv_file_path}  \n")

    return model

import torchvision.models as models
import torch
from .models.fpcn import FPCN
from .submodules import attention
from .submodules.backboned_unet import backboned_unet as unet 
import pandas as pd
import os
from .metrics import calculate
from .losses.focal import FocalLoss

def test(args, test_loader, weights=None):
    mode = 'binary' if args.num_classes == 2 else 'multiclass'
    
    test_loss = 0
    if args.TASK == 'Classify':
        if mode == 'multiclass':
            test_metrics = {
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
            test_metrics = {
                'accuracy_score': 0.0,
                'roc_auc_score': 0.0,
                'f1_score': 0.0,
            }
    elif args.TASK == 'Segment':
            test_metrics = {        
                'accuracy_score': 0.0,
                'dice_score': 0.0,
                'roc_auc_score': 0.0
    }

    logs_dir = os.path.join("training_runs", args.DATE)
    # Load the model
    model_path = f"{logs_dir}/best_val_acc_model.pth"
    
    
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
        else:
            raise ValueError("Please select a valid model for Segmentification problem.")

    dict = torch.load(model_path)
    # Assuming model is your PyTorch model
    print("Model State Dict Keys:")
    print(model.state_dict().keys())

    # Assuming dict is the loaded state_dict
    print("Loaded State Dict Keys:")
    print(dict.keys())
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()
    
    device = args.TO
    model = model.to(device)

    weights = torch.tensor(weights).to(device)

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


    for batch, item in enumerate(test_loader):
                    
        X = item['image']
        y = item['label']
        m_ = item['mask'].to(device)

        if args.TASK == 'Segment':
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
                test_metrics = {k: v + calculate.calculate_clf_metrics(y, yhat, mode, device)[k] for k, v in test_metrics.items()}
            elif args.TASK == 'Segment':
                loss = criterion(yhat, m)
                test_metrics = {k: v + calculate.calculate_seg_metrics(m, yhat, mode, device)[k] for k, v in test_metrics.items()}

            else:
                raise NotImplementedError
            
            test_loss += loss.item()


    test_loss /= len(test_loader)
    test_metrics = {k: v / len(test_loader) for k, v in test_metrics.items()}

    # Save results to CSV
    results_df = pd.DataFrame(test_metrics)
    results_df.to_csv(f'{logs_dir}/test_result.csv', index=False)
    print("Evaluation results saved to 'test_result.csv'.")


# Example usage:
# args = YourArgumentsObject  # Replace with your actual arguments
# test_loader = YourTestDataLoader  # Replace with your actual test loader
# Evaluate_model(args, test_loader)

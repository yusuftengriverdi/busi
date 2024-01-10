# Inside run.py
import torch
from phases.prepare import get_data_loaders
from phases.train import train
# from phases.test import test
from phases.augment import augment

from utilities import parse_arguments
import os
from datetime import datetime

# Specify the arguments as needed
args = parse_arguments()

if args.USE_MASK: args.ATTENTION = True

# Set date as unique id.
args.DATE = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create a directory to store logs
logs_dir = os.path.join("training_runs", args.DATE)

os.makedirs(logs_dir, exist_ok=True)

# Log file path
log_file_path = os.path.join(logs_dir, "log.txt")

# Set log file path as an argument so we can access from everywhere.
args.LOG = log_file_path

# Write information to the log file
with open(log_file_path, mode='a') as log_file:
    log_file.write("Arguments:\n")
    for key, value in vars(args).items():
        log_file.write(f"{key}: {value}\n")
    log_file.write("\n")
    log_file.write("General Info: Hello, world!! \n")
    log_file.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")


# Get loaders from prepare.py
train_loader, val_loader, test_loader, weights = get_data_loaders(args)

# Now you can use train_loader, val_loader, and test_loader in the rest of run.py

if args.INCLUDE_NORMAL: args.num_classes = 3
else: args.num_classes = 2

if not args.MODEL in ['Resnet18', 'Unet']: raise NotImplementedError
if not args.TASK in ['Classify', 'Segment']: raise NotImplementedError

if args.AUG:
    train_loader = augment(args, train_loader, weights)


model = train(args, train_loader = train_loader, val_loader = val_loader, weights = weights)

# Save the model
torch.save(model.state_dict(), f'{logs_dir}/model.pth')
# test(args, test_loader, model)

# Best results classification so far.

# python run.py --LR 0.01 --PRETRAINED --PREP --LOSS Focal --INCLUDE_NORMAL --EP 20 --SPLIT_RATIO 0.7:0.2:0.1 --ROOT data/ --TO cuda --ATTENTION

# Best results segmentation so far.

# python run.py --LR 0.01 --PRETRAINED --PREP --LOSS Dice --MODEL Unet --TASK Segment --INCLUDE_NORMAL --EP 20 --SPLIT_RATIO 0.7:0.2:0.1 --ROOT data/ --TO cuda
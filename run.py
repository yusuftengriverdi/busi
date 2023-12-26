# Inside run.py

from phases.prepare import get_data_loaders
from phases.train import train
# from phases.test import test

from utilities import parse_arguments
import os
from datetime import datetime
import torch

# Specify the arguments as needed
args = parse_arguments()

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

if args.MODEL != 'Resnet18': raise NotImplementedError
if args.TASK != 'Classification': raise NotImplementedError

model = train(args, train_loader = train_loader, val_loader = val_loader, weights = weights)

# Save the model
torch.save(model.state_dict(), f'{logs_dir}/model.pth')
# test(model, args)
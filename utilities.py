import argparse

def parse_arguments():
    """
    Parses command-line arguments using argparse.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Custom Dataset Preparation Script")

    # Required arguments
    parser.add_argument("--ROOT", type=str, help="Root path of the dataset", required=True)
    parser.add_argument("--MODE", type=str, help="Mode (0/1/2)", required=True)

    # Optional arguments with default values
    parser.add_argument("--INCLUDE_NORMAL", action="store_true", help="Include normal cases", default=False)
    parser.add_argument("--WIDTH", type=int, help="Width for image resizing", default=256)
    parser.add_argument("--HEIGHT", type=int, help="Height for image resizing", default=256)

    # Split arguments
    parser.add_argument("--SPLIT_RATIO", type=str, help="Split ratio for train-val-test as x:y:z", default=None)
    parser.add_argument("--SEED", type=int, help="Random seed for data splitting", default=None)

    # Batch size
    parser.add_argument("--BATCH_SIZE", type=int, help="Batch size for DataLoader", default=32)

    # Model arguments    
    parser.add_argument("--MODEL", type=str, help="Model type, Resnet18 supported", default= "Resnet18")
    parser.add_argument("--PRETRAINED", action="store_true", help="Pretrained or not", default=False)

    # Task argument
    parser.add_argument("--TASK", type=str, help="Classification, Segmentation, or both.", default= "Classification")

    # Optimizer argument
    parser.add_argument("--OPT", type=str, help="Optimizer type.", default= "SGD")
    parser.add_argument("--LR", type=float, help="Learning rate (static)", default= 1e-4)
    parser.add_argument("--MOMENTUM", type=float, help="Momentum", default= 0.9)

    # Training argument
    parser.add_argument("--LOSS", type=str, help="Optimizer type.", default= "CrossEntropy")
    parser.add_argument("--EP", type=int, help="Number of Epochs", default= 20)

    # Device argument
    parser.add_argument("--TO", type=str, help="Device type (cuda or cpu).", default= "cpu")

    return parser.parse_args()
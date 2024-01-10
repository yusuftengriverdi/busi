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

    # Data-related arguments.
    parser.add_argument("--MODE", type=int, help="Mode (0/1/2)", required=False, default=1)
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
    parser.add_argument("--TASK", type=str, help="Classify, Segment, or both.", default= "Classify")

    # Optimizer arguments
    parser.add_argument("--OPT", type=str, help="Optimizer type.", default= "SGD")
    parser.add_argument("--LR", type=float, help="Learning rate (static)", default= 1e-4)
    parser.add_argument("--MOMENTUM", type=float, help="Momentum", default= 0.9)

    # Training arguments
    parser.add_argument("--LOSS", type=str, help="Optimizer type.", default= "CrossEntropy")
    parser.add_argument("--EP", type=int, help="Number of Epochs", default= 20)

    # Device argument
    parser.add_argument("--TO", type=str, help="Device type (cuda or cpu).", default= "cpu")

    # Augmentation argument
    parser.add_argument("--AUG", action="store_true", help="Use augmentation or not", default=False)

    # Preprocessing argument
    parser.add_argument("--PREP", action="store_true", help="Use preprocessing or not", default=False)

    # Attention argument.
    parser.add_argument("--ATTENTION", action="store_true", help="Use spatial attention layer or not", default=False)
    parser.add_argument("--USE_MASK", action="store_true", help="Use spatial attention layer or not", default=False)

    # Segmentation arguments.
    parser.add_argument("--BACKBONE", type=str, help="Backbone model type, resnet18 supported, please use lower case", default= "resnet18")

    return parser.parse_args()
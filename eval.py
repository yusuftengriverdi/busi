import os
import csv
import torch
from torchvision import transforms as T
import pandas as pd 
import cv2
from tqdm import tqdm
from phases.submodules import attention

def calculate_mean_std(cache_file="mean_std_cache.pth"):
    if os.path.exists(cache_file):
        # Load cached mean and std from file
        mean_std_dict = torch.load(cache_file)
        mean = mean_std_dict["mean"]
        std = mean_std_dict["std"]
    else:
        # Calculate mean and std
        means = []
        stds = []
        for item in tqdm(desc="Calculating mean and std"):
            img = item["image"]
            means.append(torch.mean(img))
            stds.append(torch.std(img))

        mean = torch.mean(torch.tensor(means))
        std = torch.mean(torch.tensor(stds))

        # Save mean and std to cache file
        mean_std_dict = {"mean": mean, "std": std}
        torch.save(mean_std_dict, cache_file)

    return mean, std


# Function to load the model state dict
def load_model_state(model, model_path):
    model.load_state_dict(torch.load(model_path), strict=False)
    return model

# Function to predict on a single image
def predict_image(model, image, pretrained):
    if pretrained: 
        # Calculate or load mean and std
        mean, std = calculate_mean_std()

        normalize = T.Normalize(mean=mean, std=std)


        # Apply any necessary pre-processing to the image
        transform = T.Compose([T.ToPILImage(),T.Resize(256), T.CenterCrop(224), T.ToTensor(), normalize])

        image = transform(image).unsqueeze(0)  # Add batch dimension

    else:
        image = T.ToTensor(image).unsqueeze(0)
    # Set the model to evaluation mode
    model.eval()

    # Make the prediction
    with torch.no_grad():
        prediction = model(image)

    return prediction

# Function to calculate metrics (customize according to your task)
def calculate_metrics(predictions, ground_truth):
    # Replace this with your metric calculation logic
    # Example: calculating accuracy for binary classification
    correct = (predictions.round() == ground_truth).sum().item()
    total = len(ground_truth)
    accuracy = correct / total
    return accuracy


# Function to process the test set and write results to a CSV file
def process_test_set(model_path, log_path, test_set_info, dataset_path, output_csv):
    # Read model type from log file
    with open(log_path, 'r') as log_file:
        lines = log_file.readlines()
        model_type = None
        for line in lines:
            if 'MODEL' in line:
                model_type = line.split(':')[1].strip()
            if 'PRETRAINED' in line:
                pretrained = True
            if 'HEIGHT' in line:
                h = int( line.split(':')[1].strip())
            if 'WIDTH' in line:
                w = int( line.split(':')[1].strip())

    if not model_type:
        raise ValueError("Model type not found in log file.")

    model = attention.resnet18attention(num_classes=3, use_mask=True, scale_dot_product=True, pretrained=pretrained)

    # Load model
    model = load_model_state(model, model_path)


    # Create CSV file to store results
    with open(output_csv, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Image Filename', 'Prediction', 'Ground Truth'])

        info = pd.read_csv(test_set_info)
        train_accuracy = 0.0
        train_len = 0

        val_accuracy = 0.0
        val_len = 0

        test_accuracy = 0.0
        test_len = 0 

        # Process each image in the test set
        for i in tqdm(range(len(info))):
            filename, set, label = tuple(info.iloc[i].values)
            image_path = os.path.join(dataset_path, label, f'{filename}.png')
            # Read the image
            image = cv2.imread(image_path)
            image = cv2.resize(image, (h, w))

            # Make prediction
            prediction = predict_image(model, image, pretrained = pretrained)

            # Calculate metrics (replace with your ground truth labels)
            label_mapping = {
                'benign': 0,
                'malignant': 1,
                'normal': 2
            }
            ground_truth = label_mapping[label]  # Replace with your ground truth labels

            pred = torch.argmax(prediction).item()
            if set == 'train':
                train_len += 1
                if ground_truth == pred:
                    train_accuracy += 1 
            if set == 'val':
                val_len += 1
                if ground_truth == pred:
                    val_accuracy += 1

            if set == 'test':
                test_len += 1
                if ground_truth == pred:
                    test_accuracy += 1 

            # Write results to CSV
            # print(prediction, torch.argmax(prediction).item())
            writer.writerow([filename, pred , ground_truth])

    print(f"Results saved to {output_csv}, Accuracy: ", [train_accuracy, train_len, val_accuracy, val_len, test_accuracy, test_len])

log_dir = 'training_runs/2024-01-14_15-29-29'

model_path = f'{log_dir}/best_val_acc_model.pth'
log_path = f'{log_dir}/log.txt'
test_set_info = f'{log_dir}/fnames.csv'
output_csv = f'{log_dir}/test_results.csv'

process_test_set(model_path, log_path, test_set_info, dataset_path='data/prep2/', output_csv=output_csv)

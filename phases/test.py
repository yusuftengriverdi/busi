import torch
import torch.nn.functional as F
import pandas as pd
import os

def test(args, test_loader):
    logs_dir = os.path.join("training_runs", args.DATE)
    # Load the model
    model_path = f"{logs_dir}/model.pth"
    model = torch.load(model_path)
    model.eval()

    # Define metrics to calculate
    metrics = {
        # Add more metrics as needed
    }

    # Initialize results dictionary
    results = {'Metric': [], 'Value': []}

    # Perform predictions and calculate metrics
    for metric_name, metric_func in metrics.items():
        metric_value = calculate_metric(model, test_loader, metric_func)
        results['Metric'].append(metric_name)
        results['Value'].append(metric_value)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{logs_dir}/test_result.csv', index=False)
    print("Evaluation results saved to 'test_result.csv'.")

def calculate_metric(model, test_loader, metric_func):
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            predictions = F.softmax(outputs, dim=1).argmax(dim=1).cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets.cpu().numpy())

    return metric_func(all_targets, all_predictions)

# Example usage:
# args = YourArgumentsObject  # Replace with your actual arguments
# test_loader = YourTestDataLoader  # Replace with your actual test loader
# evaluate_model(args, test_loader)

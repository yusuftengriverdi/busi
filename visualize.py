import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read CSV file
n = 20
# df = pd.read_csv('training_runs/2024-01-13_17-07-58/metrics.csv')[:n]
# df2 = pd.read_csv('training_runs/2024-01-13_17-18-42/metrics.csv')[:n]
# df3 = pd.read_csv('training_runs/2024-01-13_17-41-29/metrics.csv')[:n]
# df4 = pd.read_csv('training_runs/2024-01-13_20-28-31/metrics.csv')[:n]
# df5 = pd.read_csv('training_runs/2024-01-13_20-29-04/metrics.csv')[:n]
# df6 = pd.read_csv('training_runs/2024-01-14_21-58-36/metrics.csv')[:n]
df = pd.read_csv('training_runs/2024-01-13_21-22-22/metrics.csv')[:n]
df1 = pd.read_csv('training_runs/2024-01-13_22-10-16/metrics.csv')[:n]

# # Convert 'Computational Time (m)' to numeric for correct sorting
# # df['Computational Time (m)'] = pd.to_numeric(df['Computational Time (m)'])
# # Add a 'Dataset' column to each DataFrame
# df6['Dataset'] = 'Baseline (v0)'
# df3['Dataset'] = 'Finetuned (v1)' 
# df4['Dataset'] = 'Finetuned w/ Soft-Attention (v2)'
# df5['Dataset'] = 'Preprocessed Input (v3)'
# df['Dataset'] = 'Focal Loss (v4)' # v5
df['Dataset'] = 'U-Net backboned'
df['Val Acc Score'] = [float((i.split('(')[-1]).split(')')[0] )for i in df.loc[:, 'Val Acc Score']]

df1['Dataset'] = 'U-Net backboned (pretrained)'
df1['Val Acc Score'] = [float((i.split('(')[-1]).split(')')[0] )for i in df1.loc[:, 'Val Acc Score']]

# Concatenate all DataFrames into a single DataFrame
# combined_df = pd.concat([df6, df3, df4, df5, df])
combined_df = pd.concat([df, df1])

# Set style
sns.set(style='whitegrid')

# Define the columns to plot
columns_to_plot = [
    'Epoch', 
    # 'Train Loss', 
    # 'Train Acc Score', 
    # 'Train ROCAUC Score',
    # 'Train f1 Score', 'Train Macro Acc Score', 'Train Macro f1 Score',
    # 'Train Macro Precision Score', 'Train Class Acc Scores', 'Train Class f1 Scores',
    # 'Val Loss', 
    'Val Acc Score',
    # 'Val ROCAUC Score', 
    # 'Val f1 Score',
    # 'Val Macro Acc Score', 
    # 'Val Macro f1 Score',
    #  'Val Macro Precision Score',
    # 'Val Class Acc Scores', 'Val Class f1 Scores', 'Computational Time (m)'
    'Dataset'
]

# Create a Seaborn-style line plot
plt.figure(figsize=(12, 8))
sns.pointplot(data=combined_df[columns_to_plot], x='Epoch', y='Val Acc Score', hue='Dataset')

# Customize the plot
plt.title('Training and Validation Metrics Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()

# Show the plot
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Metrics - taken from the results of the models
models = ['MLP', 'GCN', 'GCN w/ Weighted Edges', 'GCN w/ Post Embeddings']
accuracies = [81.3, 97.6, 99.7, 94.1]

# Colors and customization
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
y_pos = np.arange(len(models))

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(y_pos, accuracies, color=colors, edgecolor='black', linewidth=1.2)

# Adding labels and title
plt.xticks(y_pos, models, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Model', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')

# Adjust y-axis range and add grid
plt.ylim(50, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.gca().set_facecolor('#f9f9f9')

# Show plot
plt.tight_layout()
plt.savefig('data/accuracy.png')

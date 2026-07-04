import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# 1. Set up the actual data for Random Forest (97.03% Accuracy | 101 records)
# Row 0: Good Standing | Row 1: At-Risk
cm = np.array([[50, 1], 
               [ 2, 48]])

# 2. Configure the plot
plt.figure(figsize=(8, 6))
sns.set_theme(style="white")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Good Standing', 'At-Risk'],
            yticklabels=['Good Standing', 'At-Risk'],
            annot_kws={"size": 16, "weight": "bold"})

plt.title('Confusion Matrix: Random Forest', fontsize=14, pad=20)
plt.ylabel('Actual Academic Status', fontsize=12)
plt.xlabel('Predicted Academic Status', fontsize=12)

# 3. Save to the reports folder with high resolution
output_path = r'c:\Users\KEVIN\Downloads\FinalFlash\reports\confusion_matrix.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()
print(f"Success! Your thesis image has been generated at: {output_path}")

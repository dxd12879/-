import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 模型指標資料
data = {
    'Model': ['Decision Tree', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [0.9915, 0.9995, 0.9894],
    'Precision': [0.91, 0.99, 0.92],
    'Recall': [0.91, 0.99, 0.92],
    'F1 Score': [0.91, 0.99, 0.92]
}

df = pd.DataFrame(data)
df.set_index('Model', inplace=True)

# 繪製圖表
plt.figure(figsize=(12, 6))
df.plot(kind='bar', figsize=(12, 6), colormap='Set2', rot=0)
plt.title('Performance Comparison of Three Classification Models', fontsize=16)
plt.ylabel('Score')
plt.ylim(0.8, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Metric', loc='lower right')
plt.tight_layout()
plt.savefig('Three_Model_Comparison_English.png')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 載入資料
file_path = "C:/Users/dxd12/OneDrive/桌面/pythom/predictive_maintenance.csv"
df = pd.read_csv(file_path)

# 設定字型避免亂碼（Windows 中文系統建議）
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 畫出故障類型分佈
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Failure Type', order=df['Failure Type'].value_counts().index, palette='Set2')
plt.title("各種故障類型的分佈")
plt.xlabel("故障類型")
plt.ylabel("數量")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

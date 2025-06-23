import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 載入資料（若你已經前面處理過 df_cleaned，可直接略過）
df = pd.read_csv("predictive_maintenance.csv")
df_cleaned = df.drop(columns=["UDI", "Product ID", "Failure Type"])
df_cleaned["Type"] = pd.factorize(df_cleaned["Type"])[0]

# 設定圖表風格
sns.set(style="white", font_scale=1.1)

# ========== 1. 皮爾森相關係數圖 ==========
plt.figure(figsize=(10, 8))
pearson_corr = df_cleaned.corr(method="pearson")
sns.heatmap(pearson_corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title(" Pearson Correlation Matrix")
plt.tight_layout()
plt.show()

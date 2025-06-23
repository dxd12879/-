import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 設定中文字型（微軟正黑體）
plt.rcParams['axes.unicode_minus'] = False         # 顯示負號

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# 讀入資料
df = pd.read_csv(r"C:\Users\dxd12\OneDrive\桌面\pythom\predictive_maintenance.csv")  # 修改為你的資料檔路徑

# 移除無關欄位
df = df.drop(columns=['UDI', 'Product ID'])

# 類別變數轉換成數值
label_encoders = {}
for col in ['Type', 'Failure Type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 分割特徵與標籤
X = df.drop(columns=['Target'])
y = df['Target']

# 分割訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

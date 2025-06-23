import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import os
import matplotlib

# 設定中文字體（以 Windows 常見字體為例）
plt.rcParams['font.family'] = 'Microsoft JhengHei'
matplotlib.rcParams['axes.unicode_minus'] = False  # 負號正常顯示

# 設定圖片儲存資料夾
save_dir = r"C:\Users\dxd12\OneDrive\桌面\pythom\圖表"
os.makedirs(save_dir, exist_ok=True)

# 載入資料集
df = pd.read_csv(r"C:\Users\dxd12\OneDrive\桌面\pythom\predictive_maintenance.csv")

# 預處理
df.drop(columns=['UDI', 'Product ID'], inplace=True)
df['Type'] = LabelEncoder().fit_transform(df['Type'])
df['Failure Type'] = df['Failure Type'].replace('No Failure', 'No_Failure')

# 特徵與標籤分離
X = df.drop('Failure Type', axis=1)
y = df['Failure Type']

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定義模型
models = {
    "決策樹 (Decision Tree)": DecisionTreeClassifier(random_state=42),
    "隨機森林 (Random Forest)": RandomForestClassifier(random_state=42),
    "梯度提升 (Gradient Boosting)": GradientBoostingClassifier(random_state=42)
}

# 抓取實際類別名稱
labels = sorted(y.unique())

# 模型訓練與評估
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # 評估
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, labels=labels, target_names=labels)
    
    print(f"\n模型：{name}")
    print("準確率：", accuracy)
    print("分類報告：\n", report)

    # 繪製混淆矩陣
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} - 混淆矩陣", fontsize=14)
    plt.xlabel("預測類別")
    plt.ylabel("實際類別")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # 儲存圖像
    filename = os.path.join(save_dir, f"{name}_confusion_matrix.png")
    plt.savefig(filename)
    plt.close()

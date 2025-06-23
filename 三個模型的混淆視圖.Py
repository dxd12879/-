import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 設定中文字體

# 1. 載入資料
df = pd.read_csv("predictive_maintenance.csv")

# 2. 預處理
df.drop(columns=["UDI", "Product ID"], inplace=True)
label_encoder = LabelEncoder()
df["Type"] = label_encoder.fit_transform(df["Type"])
df["Failure Type"] = df["Failure Type"].replace("No Failure", "No_Failure")

# 特徵與目標
X = df.drop(columns=["Target", "Failure Type"])
y = df["Failure Type"]

# 切分資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 模型列表
models = {
    "決策樹 - Decision Tree": DecisionTreeClassifier(random_state=0),
    "隨機森林 - Random Forest": RandomForestClassifier(random_state=0),
    "梯度提升 - Gradient Boosting": GradientBoostingClassifier(random_state=0)
}

# 類別名稱（順序需與混淆矩陣一致）
labels = ['Heat Dissipation Failure', 'No_Failure', 'Overstrain Failure', 'Power Failure', 'Random Failure']

# 畫出混淆矩陣圖
for name, model in models.items():
    model.fit(X_train, y_train := y[y.index.isin(X_train.index)])
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"{name} - 混淆矩陣", fontsize=14)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.show()

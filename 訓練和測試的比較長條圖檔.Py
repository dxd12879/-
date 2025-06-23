import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 讀取資料
df = pd.read_csv("predictive_maintenance.csv")
df.drop(columns=['UDI', 'Product ID'], inplace=True)
df['Type'] = LabelEncoder().fit_transform(df['Type'])
df['Failure Type'] = df['Failure Type'].replace('No Failure', 'No_Failure')

X = df.drop('Failure Type', axis=1)
y = df['Failure Type']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

train_scores, test_scores = [], []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_scores.append({
        "Model": name,
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "Precision": precision_score(y_train, y_train_pred, average="weighted"),
        "Recall": recall_score(y_train, y_train_pred, average="weighted"),
        "F1 Score": f1_score(y_train, y_train_pred, average="weighted")
    })

    test_scores.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "Precision": precision_score(y_test, y_test_pred, average="weighted"),
        "Recall": recall_score(y_test, y_test_pred, average="weighted"),
        "F1 Score": f1_score(y_test, y_test_pred, average="weighted")
    })

# 合併成一個 dataframe 並繪圖
train_df = pd.DataFrame(train_scores)
train_df["Dataset"] = "Train"
test_df = pd.DataFrame(test_scores)
test_df["Dataset"] = "Test"
combined_df = pd.concat([train_df, test_df])

melted = pd.melt(combined_df, id_vars=["Model", "Dataset"], 
                 value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
                 var_name="Metric", value_name="Score")

plt.figure(figsize=(14, 6))
sns.barplot(data=melted, x="Model", y="Score", hue="Dataset", palette="Set2")
plt.title("Performance Comparison: Train vs Test")
plt.ylim(0.8, 1.05)
plt.tight_layout()
plt.show()

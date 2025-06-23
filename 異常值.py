import pandas as pd

# 讀取資料（請改成你自己的檔案路徑）
df = pd.read_csv(r"C:\Users\dxd12\OneDrive\桌面\pythom\predictive_maintenance.csv")

# 只選擇數值型欄位
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns

# 儲存異常值索引
all_outliers = pd.DataFrame()

# 每個數值欄位都跑一次 IQR 判斷
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 找出異常值
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    # 存下欄位名稱 + 異常值筆數
    print(f"{col} 的異常值數量：{len(outliers)}")
    
    # 加入總表（合併所有異常值）
    all_outliers = pd.concat([all_outliers, outliers])

# 移除重複的異常值列（可能同時多個欄位異常）
all_outliers = all_outliers.drop_duplicates()

# 印出總異常值數量與前幾筆
print(f"\n總異常值列數（去重後）：{len(all_outliers)}")
print(all_outliers.head())
# 從原始資料中移除異常值列
df_cleaned = df.drop(all_outliers.index)

# 看移除後還剩幾筆
print(f"移除異常值後的資料筆數：{len(df_cleaned)}")

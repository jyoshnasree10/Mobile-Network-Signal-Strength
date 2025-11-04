import pandas as pd
from sklearn.preprocessing import LabelEncoder
# Load dataset
df = pd.read_csv("D:/2 year/DAV 2-1/Project/signal_metrics.csv")
# 1️⃣ Drop timestamp (not needed for ML)
df.drop(columns=["Timestamp"], inplace=True)
# 2️⃣ Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
# 3️⃣ Check missing values
print("Missing values per column:\n", df.isnull().sum())
# 4️⃣ Handle missing values (if any)
df.fillna(df.mean(numeric_only=True), inplace=True)
# 5️⃣ Display encoded data info
print("\nData types after encoding:\n", df.dtypes)
print("\nPreview of preprocessed data:\n", df.head())
# Save cleaned dataset
df.to_csv("signal_metrics_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved as 'signal_metrics_cleaned.csv'")

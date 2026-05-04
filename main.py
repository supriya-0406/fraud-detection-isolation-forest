import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("transactions.csv")

print("Dataset Preview:")
print(df.head())

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Apply Isolation Forest
model = IsolationForest(contamination=0.3, random_state=42)
df['Anomaly'] = model.fit_predict(X_scaled)

# Convert output (-1 → fraud, 1 → normal)
df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})

print("\nDetection Results:")
print(df)

# Plot anomalies
plt.scatter(df['Amount'], df['Time'], c=df['Anomaly'])
plt.xlabel("Transaction Amount")
plt.ylabel("Time")
plt.title("Fraud Detection using Isolation Forest")
plt.show()

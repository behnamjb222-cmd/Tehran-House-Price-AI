import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor # This is the artificial brain
from sklearn.preprocessing import StandardScaler # This is the tool for scaling numbers
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load and combine data
print("⏳ Starting the deep learning engine...")
try:
    df2023 = pd.read_csv('2023_Tehran_House_Price.csv')
    df2024 = pd.read_csv('2024_Tehran_House_Price.csv')
    df = pd.concat([df2023, df2024], ignore_index=True)
except:
    print("❌ Files not found!")
    exit()

# 2. Cleaning (same as before)
# Note: 'Test' below replaces the Farsi word for test. Ensure your data matches this string if needed.
df = df[~df['Address'].str.contains('Test', na=False)]
df = df[(df['Meter'] >= 30) & (df['Meter'] <= 500)]
df['Price_Billion'] = df['Price'] / 10_000_000_000
df = df[(df['Price_Billion'] > 0.2) & (df['Price_Billion'] < 200)]

# 3. Advanced Preparation (Encoding)
top_regions = df['Region'].value_counts().head(50).index
df_filtered = df[df['Region'].isin(top_regions)].copy()
df_encoded = pd.get_dummies(df_filtered, columns=['Region'], drop_first=True)

features = ['Meter', 'Age', 'Rooms', 'Parking', 'Elevator'] + [col for col in df_encoded.columns if 'Region_' in col]
X = df_encoded[features]
y = df_encoded['Price_Billion']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- New and vital section: Standardization (Scaling) ---
# Making numbers understandable for the neural network
print("⚖️ Scaling the numbers...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Fitting on training data (learning mean and standard deviation)
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# Scaling test data with the same scale
X_test_scaled = scaler_X.transform(X_test)

# 5. Building the Artificial Brain (Neural Network)
print("🧠 Building and training the neural network (this might take a while)...")
# hidden_layer_sizes=(100, 50) means:
# First layer: 100 neurons (brain cells)
# Second layer: 50 neurons
# max_iter=500: Iterate through the lessons 500 times
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

# 6. Final Exam
print("📝 Grading the exam paper...")
y_pred_scaled = model.predict(X_test_scaled)
# We must convert numbers back from standard to billion scale (Inverse Transform)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

r2 = r2_score(y_test, y_pred) * 100
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "*"*40)
print(f"🔥 Neural Network Intelligence Score (R2): {r2:.2f} out of 100")
print(f"📉 Model Error: {mae:.2f} Billion Tomans")
print("*"*40)

# Comparison with the previous model
if r2 > 79.0:
    print("✅ Awesome! The neural network performed smarter than the previous linear model.")
else:
    print("⚠️ Note: Sometimes linear models work better on simple data (or we need to change layers).")

import joblib

# Saving the model brain and scalers
print("💾 Saving the model...")
joblib.dump(model, 'tehran_house_model.pkl')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ Your model was saved in 'tehran_house_model.pkl'. This file is your entire AI!")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor # این همون مغز مصنوعی است
from sklearn.preprocessing import StandardScaler # اینم ابزار کوچک‌سازی اعداد
from sklearn.metrics import r2_score, mean_absolute_error

# 1. لود و ترکیب دیتا
print("⏳ استارت موتور دیپ لرنینگ...")
try:
    df2023 = pd.read_csv('2023_Tehran_House_Price.csv')
    df2024 = pd.read_csv('2024_Tehran_House_Price.csv')
    df = pd.concat([df2023, df2024], ignore_index=True)
except:
    print("❌ فایل‌ها نیستن!")
    exit()

# 2. تمیزکاری (مثل قبل)
df = df[~df['Address'].str.contains('تست', na=False)]
df = df[(df['Meter'] >= 30) & (df['Meter'] <= 500)]
df['Price_Billion'] = df['Price'] / 10_000_000_000
df = df[(df['Price_Billion'] > 0.2) & (df['Price_Billion'] < 200)]

# 3. آماده‌سازی پیشرفته (Encoding)
top_regions = df['Region'].value_counts().head(50).index
df_filtered = df[df['Region'].isin(top_regions)].copy()
df_encoded = pd.get_dummies(df_filtered, columns=['Region'], drop_first=True)

features = ['Meter', 'Age', 'Rooms', 'Parking', 'Elevator'] + [col for col in df_encoded.columns if 'Region_' in col]
X = df_encoded[features]
y = df_encoded['Price_Billion']

# 4. تقسیم داده‌ها
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- بخش جدید و حیاتی: استانداردسازی (Scaling) ---
# اعداد رو برای شبکه عصبی قابل فهم می‌کنیم
print("⚖️ در حال استاندارد کردن اعداد (Scaling)...")
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# فیت کردن روی داده‌های آموزش (یادگیری میانگین و انحراف معیار)
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

# داده‌های تست رو هم با همون مقیاس کوچک می‌کنیم
X_test_scaled = scaler_X.transform(X_test)

# 5. ساخت مغز مصنوعی (Neural Network)
print("🧠 در حال ساخت و آموزش شبکه عصبی (این ممکنه کمی طول بکشه)...")
# hidden_layer_sizes=(100, 50) یعنی:
# لایه اول: ۱۰۰ تا نورون (سلول مغزی)
# لایه دوم: ۵۰ تا نورون
# max_iter=500: یعنی ۵۰۰ بار کل درس‌ها رو مرور کن
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train_scaled)

# 6. امتحان نهایی
print("📝 در حال تصحیح برگه امتحان...")
y_pred_scaled = model.predict(X_test_scaled)
# باید اعداد رو از حالت استاندارد برگردونیم به حالت میلیاردی (Inverse Transform)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

r2 = r2_score(y_test, y_pred) * 100
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "*"*40)
print(f"🔥 نمره هوش شبکه عصبی (R2): {r2:.2f} از ۱۰۰")
print(f"📉 خطای مدل: {mae:.2f} میلیارد تومان")
print("*"*40)

# مقایسه با مدل قبلی
if r2 > 79.0:
    print("✅ ایول! شبکه عصبی از مدل خطی قبلی باهوش‌تر عمل کرد.")
else:
    print("⚠️ نکته: گاهی روی داده‌های ساده، مدل خطی بهتر جواب میده (یا باید لایه‌ها رو تغییر بدیم).")

import joblib

# ذخیره کردن مغز مدل (model) و مترجم‌ها (scaler)
print("💾 در حال ذخیره سازی مدل...")
joblib.dump(model, 'tehran_house_model.pkl')
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')
print("✅ مدل شما در فایل 'tehran_house_model.pkl' ذخیره شد. این فایل یعنی تمام هوش مصنوعی شما!")
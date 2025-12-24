import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ==========================================
# 1. Load Model and Settings (Backend)
# ==========================================
# Cache model for speed (so it doesn't reload on every refresh)
@st.cache_resource
def load_models():
    try:
        model = joblib.load('tehran_house_model.pkl')
        scaler_X = joblib.load('scaler_X.pkl')
        scaler_y = joblib.load('scaler_y.pkl')
        return model, scaler_X, scaler_y
    except FileNotFoundError:
        return None, None, None


# Load files
model, scaler_X, scaler_y = load_models()

# List of regions (must match exactly what the model saw during training)
# Here are some examples, but in reality, all regions should be listed
regions_list = [
    'Punak', 'Saadat Abad', 'Tajrish', 'Velenjak', 'Zafaranieh', 'Niavaran',
    'Shahrak Gharb', 'Janat Abad', 'Pasdaran', 'Tehranpars', 'Sattarkhan', 'Vanak'
]

# ==========================================
# 2. UI Design (Frontend)
# ==========================================
# Page settings
st.set_page_config(page_title="Tehran House Price AI", page_icon="🏠")

# Main title
st.title("🏠 Tehran House Price Prediction (AI)")
st.write("This tool uses **Deep Learning** to estimate real estate prices.")
st.divider()  # Separator line

# Columns (two columns side by side for inputs)
col1, col2 = st.columns(2)

with col1:
    meter = st.number_input("Area (sq. meters)", min_value=30, max_value=500, value=100)
    age = st.slider("Building Age (years)", 0, 30, 5)
    region = st.selectbox("District", regions_list)

with col2:
    rooms = st.slider("Room Count", 1, 5, 2)
    parking = st.checkbox("Has Parking?", value=True)
    elevator = st.checkbox("Has Elevator?", value=True)

# Action button
if st.button("💰 Calculate Price", type="primary"):

    if model is None:
        st.error("❌ Model files not found! Make sure .pkl files are next to app.py.")
    else:
        # ==========================================
        # 3. Prediction Logic
        # ==========================================
        # Convert user inputs to numbers (1 and 0)
        park_val = 1 if parking else 0
        elev_val = 1 if elevator else 0

        # Create a raw data row (like we did in the previous file)
        # Note: Here columns structure must match training exactly
        # We take a shortcut:

        # 1. Extract column names expected by the model (from scaler)
        expected_features = scaler_X.feature_names_in_

        # 2. Create a row with all zeros
        input_data = pd.DataFrame(0, index=[0], columns=expected_features)

        # 3. Fill main values
        input_data['Meter'] = meter
        input_data['Age'] = age
        input_data['Rooms'] = rooms
        input_data['Parking'] = park_val
        input_data['Elevator'] = elev_val

        # 4. Fill region (Manual One-Hot Encoding)
        # NOTE: If your model was trained on Farsi names, you might need to map these English names back to Farsi here.
        region_col = f"Region_{region}"
        if region_col in input_data.columns:
            input_data[region_col] = 1

        # 5. Standardization (Translate for the model brain)
        input_scaled = scaler_X.transform(input_data)

        # 6. Prediction
        pred_scaled = model.predict(input_scaled)

        # 7. Convert price back to original scale (Tomans/Billions)
        price_billion = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

        # Display result
        st.success(f"Estimated Price: {price_billion:.2f} Billion Tomans")

        # Calculate price per meter
        price_per_meter = (price_billion * 1000) / meter
        st.info(f"Price per meter: {price_per_meter:.1f} Million Tomans")
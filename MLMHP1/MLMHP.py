import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# --- Keele University Logo (Centered) ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("keele_logo.jpg", use_column_width=True)

# --- Load trained model and objects ---
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("postcode_encoder.pkl", "rb") as f:
    postcode_encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("poly_transform.pkl", "rb") as f:
    poly = pickle.load(f)

with open("column_order.pkl", "rb") as f:
    column_order = pickle.load(f)

# --- Load postcode data ---
df_postcode = pd.read_csv("m41_prices_2015_to_2023.csv")
df_postcode["postcode_prefix"] = df_postcode["postcode"].str.extract(r"(M\d{2}\s?\d)").fillna("M41 0")
df_postcode["postcode_prefix_encoded"] = postcode_encoder.transform(df_postcode["postcode_prefix"].astype(str))

# Create a pseudo floor_area estimate (same logic as training)
df_postcode["floor_area"] = np.select(
    [
        df_postcode["property_type"] == "F",
        df_postcode["property_type"] == "T",
        df_postcode["property_type"] == "S",
        df_postcode["property_type"] == "D"
    ],
    [55, 80, 100, 130],
    default=90
)

# --- Prepare postcode stats ---
df_postcode["price_per_m2"] = df_postcode["price"] / df_postcode["floor_area"]
postcode_stats = df_postcode[df_postcode["date_of_transfer"].str.startswith("2023")].copy()
postcode_agg = postcode_stats.groupby("postcode_prefix_encoded").agg({
    "price": "mean",
    "price_per_m2": "mean"
}).rename(columns={
    "price": "postcode_avg_price",
    "price_per_m2": "price_per_m2"
}).reset_index()

# --- UI Layout ---
st.title("M41 House Price Predictor")
st.caption("Trained on 9 years of Land Registry data (2015–2023) - Created by Max")

# --- Inputs ---
property_type = st.selectbox("Property Type", ["Flat", "Terraced", "Semi-detached", "Detached"])
floor_area = st.slider("Estimated Floor Area (m²)", 30, 200, 90)
property_age = st.slider("Estimated Property Age (Years)", 0, 100, 40)
postcode = st.selectbox("Postcode Prefix", sorted(df_postcode["postcode_prefix"].unique()))
tenure = st.selectbox("Tenure Type", ["Freehold", "Leasehold"])
has_garden = st.checkbox("Has a Garden", value=True)
bedrooms = st.slider("Number of Bedrooms", 1, 6, 3)
year_sold = st.slider("Year of Sale", 2015, 2024, 2023)
ppd_category = st.radio("PPD Category", ["A - Standard market transaction", "B - Other"])
record_status = st.radio("Record Status", ["C - Change to record", "A - New Sale"])

# --- Mapping & Derived Features ---
property_type_map = {"Flat": 0, "Terraced": 1, "Semi-detached": 2, "Detached": 3}
tenure_map = {"Freehold": 1, "Leasehold": 0}
postcode_encoded = postcode_encoder.transform([postcode])[0]
postcode_row = postcode_agg[postcode_agg["postcode_prefix_encoded"] == postcode_encoded]

price_per_m2 = postcode_row["price_per_m2"].values[0] if not postcode_row.empty else 3000
postcode_avg_price = postcode_row["postcode_avg_price"].values[0] if not postcode_row.empty else 300000

is_old_build = int(property_age > 30)
age_area_combo = property_age * floor_area
ppd_category_val = 0 if ppd_category == "B" else 1
record_status_val = 0 if record_status == "C" else 1

# --- Prepare input dataframe ---
input_features = {
    "property_type": property_type_map[property_type],
    "new_build": 0,
    "tenure": tenure_map[tenure],
    "locality": 0,
    "town_city": 0,
    "district": 0,
    "county": 0,
    "year_sold": year_sold,
    "floor_area": floor_area,
    "property_age": property_age,
    "price_per_m2": price_per_m2,
    "postcode_prefix_encoded": postcode_encoded,
    "has_garden": int(has_garden),
    "bedrooms": bedrooms,
    "is_old_build": is_old_build,
    "postcode_avg_price": postcode_avg_price,
    "age_area_combo": age_area_combo,
    "ppd_category_type": ppd_category_val,
    "record_status": record_status_val
}
input_df = pd.DataFrame([input_features])
input_df = input_df[column_order]  # Ensure correct order

# --- Prediction ---
# --- Optional Multipliers ---
st.markdown("### Additional Features")
has_driveway = st.checkbox("Off-road Parking / Driveway")
has_garage = st.checkbox("Garage")
condition = st.selectbox("Condition of Property", ["Excellent", "Good", "Satisfactory", "Poor / In Need of Repair"])

# --- Prediction ---
if st.button("Predict House Price"):
    X_scaled = scaler.transform(input_df)
    X_poly = poly.transform(X_scaled)
    prediction = model.predict(X_poly)[0]

    # Apply multipliers
    multiplier = 1.0
    if has_driveway:
        multiplier *= 1.055
    if has_garage:
        multiplier *= 1.06
    condition_multiplier = {
        "Excellent": 1.05,
        "Good": 1.03,
        "Satisfactory": 1.00,
        "Poor / In Need of Repair": 0.90
    }
    multiplier *= condition_multiplier[condition]

    adjusted_prediction = prediction * multiplier

    # Confidence margin
    mae_buffer = 5000
    lower = adjusted_prediction - mae_buffer
    upper = adjusted_prediction + mae_buffer

    # Display results
    st.subheader("Predicted Sale Price:")
    st.write(f"£{adjusted_prediction:,.0f}")
    st.caption(f"Estimated range: £{lower:,.0f} – £{upper:,.0f}")

    st.subheader("Postcode Average:")
    st.write(f"£{postcode_avg_price:,.0f}")


   # --- Optional: Model Performance Display ---
with st.expander("Model Accuracy"):
    st.write("This model achieved:")
    st.markdown("- **R² Score** ≈ 0.998 - Can be seen as the model's accuracy")
    st.markdown("- **Mean Absolute Error** ≈ £1000-£5000 - How far off the model predicts, when the model is wrong")





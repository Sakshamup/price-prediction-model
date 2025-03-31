import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Flight Price Predictor ✈️", layout="wide", page_icon="✈️")

st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stSuccess {
        color: #4CAF50;
        font-size: 18px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #2E86C1;
    }
    .stDataFrame {
        border-radius: 8px;
        box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

model = joblib.load("xgboost_flight_model.pkl")

try:
    transformer = joblib.load("transformer.pkl")
    if not hasattr(transformer, "transform"):
        st.error("Loaded transformer is not valid! Check transformer.pkl.")
        transformer = None
except Exception as e:
    st.error(f"Error loading transformer: {e}")
    transformer = None

def clean_feature_names(feature_names):
    return [name.replace("tf", "").replace("_", " ").strip().lstrip("0123456789_") for name in feature_names]

def main():
    st.title("Flight Price Prediction with Explainable AI 🤖")
    st.markdown("### Predict flight prices and understand model decisions!")

    st.sidebar.image("john-mcarthur-TWBkfxTQin8-unsplash.jpg", width=250)
    st.sidebar.write("## Input Flight Details")

    with st.sidebar.expander("Flight Details", expanded=True):
        source_city = st.selectbox("Source City", ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'])
        destination_city = st.selectbox("Destination City", ['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Kolkata'])
        airline = st.selectbox("Airline", ['Air India', 'IndiGo', 'SpiceJet', 'Vistara', 'Go First'])
        departure_time = st.selectbox("Departure Time", ['Morning', 'Afternoon', 'Evening', 'Night'])
        arrival_time = st.selectbox("Arrival Time", ['Morning', 'Afternoon', 'Evening', 'Night'])
        flight_class = st.radio("Class", ['Economy', 'Business'])
        stops = st.radio("Number of Stops", ['zero', 'one', 'two_or_more'])
        duration = st.slider("Flight Duration (hours)", 1, 15, 2)
        days_left = st.slider("Days Left to Departure", 0, 90, 30)

    if st.sidebar.button("Predict Price ✈️"):
        input_data = pd.DataFrame({
            'source_city': [source_city],
            'destination_city': [destination_city],
            'airline': [airline],
            'departure_time': [departure_time],
            'arrival_time': [arrival_time],
            'class': [flight_class],
            'stops': [stops],
            'duration': [duration],
            'days_left': [days_left]
        })

        if transformer is not None:
            try:
                input_transformed = transformer.transform(input_data)
            except Exception as e:
                st.error(f"Error transforming input data: {e}")
                return
        else:
            st.error("Transformer is missing. Check transformer.pkl.")
            return

        predicted_price = model.predict(input_transformed)[0]
        st.success(f"Predicted Flight Price: ₹{predicted_price:.2f}")
        st.balloons()

        st.subheader("🔍 Why Did the Model Predict This Price?")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_transformed)

        feature_names = (transformer.get_feature_names_out() if hasattr(transformer, "get_feature_names_out") else input_data.columns)
        clean_names = clean_feature_names(feature_names)

        shap_df = pd.DataFrame({
            "Feature": clean_names,
            "SHAP Value": shap_values[0]
        }).sort_values(by="SHAP Value", ascending=False)

        shap_df["Contribution"] = shap_df["SHAP Value"].apply(lambda x: "Increases Price" if x > 0 else "Decreases Price")

        st.dataframe(shap_df)

        st.subheader("Key Factors Influencing the Predicted Price")
        for i, row in shap_df.iterrows():
            feature = row["Feature"]
            value = input_data.iloc[0].get(feature.replace(" ", "_"), "N/A")
            st.write(f"- **{feature}**: {value} ({row['Contribution']} by ₹{abs(row['SHAP Value']):.2f})")

        st.subheader("Visual Explanation (SHAP Waterfall Plot)")
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, feature_names=clean_names), max_display=5, show=False)
        st.pyplot(fig)

        st.subheader("Overall Feature Importance (Summary Plot)")
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, input_transformed, feature_names=clean_names, show=False)
        st.pyplot(fig)

if __name__ == "__main__":
    main()

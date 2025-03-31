import streamlit as st
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image  # For adding images

# Set page config at the top
st.set_page_config(page_title="🏡 Housing Price Prediction App", layout="wide", page_icon="🏡")

# Custom CSS for styling
st.markdown("""
    <style>
    /* Change the background color of the main content area */
    .stApp {
        background-color: #000000;  /* Black background */
        color: white;
    }

    /* Change the background color of the sidebar */
    .stSidebar {
        background-color: #1c1c1c;  /* Dark gray for the sidebar */
        padding: 20px;
        border-radius: 8px;
    }

    /* Change the background color of expanders */
    .stExpander {
        background-color: #1c1c1c;  /* Dark gray background for expanders */
        border-radius: 8px;
        box-shadow: 0 4px 8px 0 rgba(255, 255, 255, 0.1);
    }

    /* Change the text color for better readability */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4A90E2;  /* Light blue for headers */
    }

    .stMarkdown p, .stMarkdown li {
        color: #ffffff;  /* White for text */
    }

    /* Style buttons */
    .stButton>button {
        background-color: #4CAF50;  /* Green button */
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #45a049;  /* Darker green on hover */
    }

    /* Style success messages */
    .stSuccess {
        color: #4CAF50;  /* Green for success messages */
        font-size: 18px;
        font-weight: bold;
    }

    /* Style dataframes */
    .stDataFrame {
        background-color: #1c1c1c;  /* Dark gray background for dataframes */
        border-radius: 8px;
        box-shadow: 0 4px 8px 0 rgba(255, 255, 255, 0.2);
        color: white;
    }

    /* Style tables inside dataframes */
    .stDataFrame table {
        background-color: #1c1c1c;  /* Dark gray background for tables */
        color: white;
    }

    /* Style images */
    .stImage {
        background-color: #000000;  /* Black background for images */
    }

    /* Change input box and select box styles */
    .stTextInput>div>div>input, 
    .stNumberInput>div>div>input, 
    .stSelectbox>div>div>select {
        background-color: #333333 !important; /* Dark background */
        color: white !important; /* White text */
        border-radius: 8px;
    }

    /* Style for checkboxes */
    .stCheckbox>div>div {
        color: white;
    }

    </style>
    """, unsafe_allow_html=True)


# Load the trained model and scaler
model = joblib.load('catboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize SHAP explainer
explainer = shap.TreeExplainer(model)

# Streamlit UI
st.title("🏡 Housing Price Prediction App")
st.markdown("### Enter the property details below to get the predicted price and understand why the model made its decision.")

# Add a banner image
banner_image = Image.open("pexels-binyaminmellish-106399.jpg")  # Replace with your image path
st.image(banner_image, use_container_width=True, caption="Find Your Dream Home")

# Sidebar for user inputs
st.sidebar.title("Property Details")
st.sidebar.markdown("### Enter the details of the property to predict its price.")

# User input fields in the sidebar
with st.sidebar.expander("Property Features", expanded=True):
    area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1500)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
    stories = st.number_input("Number of Stories", min_value=1, max_value=5, value=1)
    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

    # Categorical inputs
    furnishingstatus = st.selectbox("Furnishing Status", ["Unfurnished", "Semi-Furnished", "Furnished"])
    mainroad = st.checkbox("Near Main Road")
    guestroom = st.checkbox("Has Guest Room")
    basement = st.checkbox("Has Basement")
    hotwaterheating = st.checkbox("Hot Water Heating")
    airconditioning = st.checkbox("Has Air Conditioning")
    prefarea = st.checkbox("Preferred Area")

# Encode categorical inputs
furnishing_dict = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
features = np.array([
    area, bedrooms, bathrooms, stories, parking,
    furnishing_dict[furnishingstatus], int(mainroad), int(guestroom), int(basement),
    int(hotwaterheating), int(airconditioning), int(prefarea)
]).reshape(1, -1)

# Scale features
features_scaled = scaler.transform(features)

# Predict price
if st.button("Predict Price 🏡"):
    price = model.predict(features_scaled)[0]
    st.success(f"🏠 Estimated Price: ₹{price:,.2f}")
    st.balloons()

    # Explainability with SHAP
    st.subheader("🔍 Why Did the Model Predict This Price?")
    shap_values = explainer.shap_values(features_scaled)

    # Feature names
    feature_names = [
        "Area", "Bedrooms", "Bathrooms", "Stories", "Parking",
        "Furnishing Status", "Near Main Road", "Has Guest Room", "Has Basement",
        "Hot Water Heating", "Has Air Conditioning", "Preferred Area"
    ]

    # Display SHAP values as a table for detailed reasoning
    st.subheader("Feature Contributions to the Predicted Price")
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP Value": shap_values[0]
    })
    shap_df = shap_df.sort_values(by="SHAP Value", ascending=False)

    # Add a column for the direction of contribution
    shap_df["Contribution"] = shap_df["SHAP Value"].apply(
        lambda x: "Increases Price" if x > 0 else "Decreases Price"
    )
    st.dataframe(shap_df, use_container_width=True)

    # Explain the top contributing features
    st.subheader("Key Factors Influencing the Predicted Price")
    for i, row in shap_df.iterrows():
        feature = row["Feature"]
        value = features_scaled[0][i]  # Scaled value
        shap_value = row["SHAP Value"]
        contribution = row["Contribution"]
        st.write(
            f"- **{feature}**: {value:.2f} "
            f"({contribution} by ₹{abs(shap_value):.2f})"
        )

    # Visualize SHAP values
    st.subheader("Visual Explanation (SHAP Waterfall Plot)")
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, feature_names=feature_names), max_display=5, show=False)
    st.pyplot(fig)

    # Summary Plot for Global Feature Importance
    st.subheader("Overall Feature Importance (Summary Plot)")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, features_scaled, feature_names=feature_names, show=False)
    st.pyplot(fig)

# Add a footer
st.markdown("---")
st.markdown("Created with ❤️ by Saksham Upadhyay")
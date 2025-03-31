import streamlit as st

# Set the page title and layout
st.set_page_config(page_title="Price Prediction Hub", layout="centered")

st.title("🔮 Price Prediction Hub")
st.markdown("### Choose a category to predict prices:")

# Display images and navigation buttons
col1, col2 = st.columns(2)

with col1:
    st.image("john-mcarthur-8KLLgqHMAv4-unsplash.jpg", use_container_width=True)  # ✅ Correct

    if st.button("✈️ Flight Price Prediction"):
        st.switch_page("pages/flight.py")  # Correct
with col2:
    st.image("todd-kent-178j8tJrNlc-unsplash.jpg", use_container_width=True)  # ✅ Correct
    if st.button("🏡 House Price Prediction"):
        st.switch_page("pages/house.py")
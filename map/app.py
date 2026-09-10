import streamlit as st
import pandas as pd
import pydeck as pdk
import joblib
import numpy as np
from geopy.geocoders import Nominatim

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="USGS Earthquake AI",
    layout="wide"
)

st.title("USGS Earthquake AI — Damage Prediction")

st.write(
    "Choose a location and enter the earthquake characteristics "
    "to estimate the expected damage level."
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

damage_model = joblib.load(
    "models/damage_random_forest.joblib"
)


# ============================================================
# SEARCH PLACE
# ============================================================

st.subheader("1. Choose a place")

geolocator = Nominatim(
    user_agent="usgs_earthquake_ai"
)

place = st.text_input(
    "Search for a city or place",
    value="Montreal, Canada"
)

location = geolocator.geocode(place)

if location:

    latitude = location.latitude
    longitude = location.longitude

    st.success(
        f"Location found: {location.address}"
    )

    col_lat, col_lon = st.columns(2)

    col_lat.metric(
        "Latitude",
        round(latitude, 4)
    )

    col_lon.metric(
        "Longitude",
        round(longitude, 4)
    )

else:

    st.error(
        "Location not found. Try another city or place."
    )

    st.stop()



# ============================================================
# EARTHQUAKE CHARACTERISTICS
# ============================================================

st.subheader("2. Enter earthquake characteristics")

col1, col2, col3 = st.columns(3)

with col1:

    mag = st.number_input(
        "Magnitude",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1
    )

    depth = st.number_input(
        "Depth (km)",
        min_value=0.0,
        value=10.0,
        step=1.0
    )

    mmi = st.number_input(
        "MMI",
        min_value=0.0,
        max_value=12.0,
        value=5.0,
        step=0.1
    )


with col2:

    cdi = st.number_input(
        "CDI",
        min_value=0.0,
        max_value=10.0,
        value=4.1,
        step=0.1
    )

    felt = st.number_input(
        "Number of felt reports",
        min_value=0,
        value=10,
        step=1
    )

    sig = st.number_input(
        "USGS Significance (SIG)",
        min_value=0,
        value=400,
        step=10
    )


with col3:

    tsunami_option = st.selectbox(
        "Tsunami",
        ["No", "Yes"]
    )

    tsunami = 1 if tsunami_option == "Yes" else 0


# ============================================================
# PREDICTION
# ============================================================

st.subheader("3. AI prediction")

if st.button("Predict Damage", type="primary"):

    # Same transformation used during training
    felt_log_imputed = np.log1p(felt)

    cdi_imputed = cdi

    features = pd.DataFrame([{
        "mag": mag,
        "depth": depth,
        "mmi": mmi,
        "cdi_imputed": cdi_imputed,
        "felt_log_imputed": felt_log_imputed,
        "sig": sig,
        "tsunami": tsunami,
        "latitude": latitude,
        "longitude": longitude
    }])

    prediction = damage_model.predict(features)[0]


    # ========================================================
    # RESULT
    # ========================================================

    st.write("### Prediction for", place)

    if prediction == "Faible":

        st.success(
            "Predicted Damage: FAIBLE"
        )

        color = [0, 180, 0, 220]


    elif prediction == "Modéré":

        st.warning(
            "Predicted Damage: MODÉRÉ"
        )

        color = [255, 165, 0, 220]


    else:

        st.error(
            "Predicted Damage: SÉVÈRE"
        )

        color = [220, 0, 0, 220]


    # ========================================================
    # MAP DATA
    # ========================================================

    map_df = pd.DataFrame([{
        "place": place,
        "latitude": latitude,
        "longitude": longitude,
        "mag": mag,
        "depth": depth,
        "mmi": mmi,
        "prediction": prediction,
        "color": color
    }])


    # ========================================================
    # MAP POINT
    # ========================================================

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position="[longitude, latitude]",
        get_fill_color="color",
        get_radius=50000,
        pickable=True,
        auto_highlight=True
    )


    view_state = pdk.ViewState(
        latitude=latitude,
        longitude=longitude,
        zoom=4
    )


    tooltip = {
        "html": """
        <b>{place}</b><br>
        Magnitude: {mag}<br>
        Depth: {depth} km<br>
        MMI: {mmi}<br>
        Predicted Damage: {prediction}
        """,
        "style": {
            "backgroundColor": "black",
            "color": "white"
        }
    }


    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style=None
    )


    st.pydeck_chart(deck)
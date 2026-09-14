import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from datetime import date
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="USGS Earthquake AI",
    layout="wide"
)

st.title("USGS Earthquake AI — Damage Prediction")

st.write(
    "Choose a date, click a location on the map, "
    "enter the earthquake characteristics, "
    "and let the AI estimate the expected damage level."
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

damage_model = joblib.load(
    "models/damage_random_forest.joblib"
)


# ============================================================
# SESSION STATE
# ============================================================

if "latitude" not in st.session_state:
    st.session_state.latitude = 45.5017

if "longitude" not in st.session_state:
    st.session_state.longitude = -73.5673

if "prediction" not in st.session_state:
    st.session_state.prediction = None


# ============================================================
# 1. CHOOSE DATE
# ============================================================

st.subheader("1. Choose a date")

selected_date = st.date_input(
    "Date",
    value=date.today()
)

st.info(
    f"Selected date: {selected_date}"
)

st.caption(
    "The selected date identifies the prediction scenario. "
    "The current Random Forest model does not use the date "
    "as an input variable."
)


# ============================================================
# 2. CLICK LOCATION ON MAP
# ============================================================

st.subheader("2. Click a location on the map")

st.write(
    "Click anywhere on the map to select the location "
    "where you want to estimate earthquake damage."
)


# ============================================================
# CREATE SELECTION MAP
# ============================================================

selection_map = folium.Map(
    location=[
        st.session_state.latitude,
        st.session_state.longitude
    ],
    zoom_start=3
)


# ============================================================
# CURRENT LOCATION CIRCLE
# ============================================================

folium.CircleMarker(
    location=[
        st.session_state.latitude,
        st.session_state.longitude
    ],
    radius=8,
    color="blue",
    fill=True,
    fill_color="blue",
    fill_opacity=0.8,
    tooltip="Selected location",
    popup=(
        f"Latitude: {st.session_state.latitude:.4f}<br>"
        f"Longitude: {st.session_state.longitude:.4f}"
    )
).add_to(selection_map)


# ============================================================
# DISPLAY CLICKABLE MAP
# ============================================================

map_data = st_folium(
    selection_map,
    width=None,
    height=500,
    key="selection_map",
    returned_objects=[
        "last_clicked"
    ]
)


# ============================================================
# READ MAP CLICK
# ============================================================

if map_data and map_data.get("last_clicked"):

    clicked_latitude = map_data["last_clicked"]["lat"]
    clicked_longitude = map_data["last_clicked"]["lng"]

    location_changed = (
        abs(
            clicked_latitude
            - st.session_state.latitude
        ) > 0.0001
        or
        abs(
            clicked_longitude
            - st.session_state.longitude
        ) > 0.0001
    )

    if location_changed:

        st.session_state.latitude = clicked_latitude
        st.session_state.longitude = clicked_longitude

        # Remove the previous prediction when location changes
        st.session_state.prediction = None

        st.rerun()


# ============================================================
# SHOW SELECTED COORDINATES
# ============================================================

col_lat, col_lon = st.columns(2)

col_lat.metric(
    "Selected Latitude",
    round(
        st.session_state.latitude,
        4
    )
)

col_lon.metric(
    "Selected Longitude",
    round(
        st.session_state.longitude,
        4
    )
)


# ============================================================
# 3. EARTHQUAKE CHARACTERISTICS
# ============================================================

st.subheader(
    "3. Enter earthquake characteristics"
)

col1, col2, col3 = st.columns(3)


# ============================================================
# COLUMN 1
# ============================================================

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
        max_value=800.0,
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


# ============================================================
# COLUMN 2
# ============================================================

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


# ============================================================
# COLUMN 3
# ============================================================

with col3:

    tsunami_option = st.selectbox(
        "Tsunami",
        [
            "No",
            "Yes"
        ]
    )

    tsunami = (
        1
        if tsunami_option == "Yes"
        else 0
    )

    st.write("Prediction location")

    st.write(
        f"Latitude: "
        f"{st.session_state.latitude:.4f}"
    )

    st.write(
        f"Longitude: "
        f"{st.session_state.longitude:.4f}"
    )


# ============================================================
# 4. AI PREDICTION
# ============================================================

st.subheader("4. AI prediction")

if st.button(
    "Predict Damage",
    type="primary"
):

    # ========================================================
    # SAME PREPROCESSING USED DURING TRAINING
    # ========================================================

    felt_log_imputed = np.log1p(
        felt
    )

    cdi_imputed = cdi


    # ========================================================
    # CREATE MODEL INPUT
    # ========================================================

    features = pd.DataFrame(
        [
            {
                "mag": mag,
                "depth": depth,
                "mmi": mmi,
                "cdi_imputed": cdi_imputed,
                "felt_log_imputed": felt_log_imputed,
                "sig": sig,
                "tsunami": tsunami,
                "latitude": st.session_state.latitude,
                "longitude": st.session_state.longitude
            }
        ]
    )


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = damage_model.predict(
        features
    )[0]

    st.session_state.prediction = prediction


# ============================================================
# SHOW PREDICTION
# ============================================================

if st.session_state.prediction is not None:

    prediction = st.session_state.prediction

    st.write(
        f"### Prediction for {selected_date}"
    )

    st.write(
        f"Location: "
        f"{st.session_state.latitude:.4f}, "
        f"{st.session_state.longitude:.4f}"
    )


    # ========================================================
    # RESULT COLOR
    # ========================================================

    if prediction == "Faible":

        st.success(
            "Predicted Damage: FAIBLE"
        )

        marker_color = "green"


    elif prediction == "Modéré":

        st.warning(
            "Predicted Damage: MODÉRÉ"
        )

        marker_color = "orange"


    else:

        st.error(
            "Predicted Damage: SÉVÈRE"
        )

        marker_color = "red"


    # ========================================================
    # RESULT MAP
    # ========================================================

    st.subheader(
        "Prediction on map"
    )

    result_map = folium.Map(
        location=[
            st.session_state.latitude,
            st.session_state.longitude
        ],
        zoom_start=6
    )


    # ========================================================
    # PREDICTION CIRCLE
    # ========================================================

    folium.CircleMarker(
        location=[
            st.session_state.latitude,
            st.session_state.longitude
        ],

        radius=15,

        color=marker_color,

        fill=True,

        fill_color=marker_color,

        fill_opacity=0.8,

        popup=folium.Popup(
            f"""
            <b>Date:</b> {selected_date}<br>
            <b>Latitude:</b> {st.session_state.latitude:.4f}<br>
            <b>Longitude:</b> {st.session_state.longitude:.4f}<br>
            <b>Magnitude:</b> {mag}<br>
            <b>Depth:</b> {depth} km<br>
            <b>MMI:</b> {mmi}<br>
            <b>CDI:</b> {cdi}<br>
            <b>Felt reports:</b> {felt}<br>
            <b>SIG:</b> {sig}<br>
            <b>Tsunami:</b> {tsunami_option}<br>
            <b>Predicted Damage:</b> {prediction}
            """,
            max_width=350
        ),

        tooltip=(
            f"Predicted Damage: {prediction}"
        )

    ).add_to(result_map)


    # ========================================================
    # DISPLAY RESULT MAP
    # ========================================================

    st_folium(
        result_map,
        width=None,
        height=500,
        key="result_map",
        returned_objects=[]
    )
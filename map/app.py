import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import requests

from datetime import date
from streamlit_folium import st_folium


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="USGS Earthquake AI",
    layout="wide"
)


# ============================================================
# LOAD DAMAGE MODEL
# ============================================================

damage_model = joblib.load(
    "models/damage_random_forest.joblib"
)


# ============================================================
# LOAD MODEL 7 — ECONOMIC LOSS
# ============================================================

economic_loss_artifact = joblib.load(
    "models/economic_loss_gradient_boosting.pkl"
)

economic_loss_model = economic_loss_artifact["model"]

economic_loss_features = economic_loss_artifact["features"]

economic_loss_target = economic_loss_artifact["target"]

economic_loss_target_transform = (
    economic_loss_artifact["target_transform"]
)

economic_loss_inverse_transform = (
    economic_loss_artifact["inverse_transform"]
)


# ============================================================
# ECONOMIC LOSS INVERSE TRANSFORMATION
# ============================================================

def inverse_economic_loss(value):
    """
    Convert the Model 7 prediction back to the
    original economic-loss scale.

    The transformation information is stored directly
    inside the Model 7 model artifact.
    """

    if callable(economic_loss_inverse_transform):

        result = economic_loss_inverse_transform(
            value
        )

        if np.isscalar(result):
            return float(result)

        return float(
            np.asarray(result).reshape(-1)[0]
        )

    transform_name = str(
        economic_loss_inverse_transform
    ).lower()

    if (
        "expm1" in transform_name
        or "log1p" in transform_name
    ):

        return float(
            np.expm1(value)
        )

    if (
        transform_name in [
            "none",
            "identity",
            "no_transform"
        ]
    ):

        return float(value)

    # Model 7 currently uses log1p.
    # Keep a safe fallback for the stored artifact.
    if (
        "log" in transform_name
    ):

        return float(
            np.expm1(value)
        )

    return float(value)


# ============================================================
# LOAD USGS EVENT
# ============================================================

def load_usgs_event(event_id):

    event_id = event_id.strip()

    if not event_id:
        raise ValueError(
            "Veuillez entrer un identifiant USGS."
        )

    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?eventid={event_id}&format=geojson"
    )

    response = requests.get(
        url,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if (
        "properties" not in data
        or "geometry" not in data
    ):
        raise ValueError(
            "Événement USGS introuvable."
        )

    properties = data["properties"]

    geometry = data["geometry"]

    coordinates = geometry.get(
        "coordinates",
        []
    )

    if len(coordinates) < 3:
        raise ValueError(
            "Les coordonnées de l'événement "
            "USGS sont indisponibles."
        )

    longitude = coordinates[0]

    latitude = coordinates[1]

    depth = coordinates[2]

    magnitude = properties.get(
        "mag"
    )

    sig = properties.get(
        "sig"
    )

    tsunami = properties.get(
        "tsunami"
    )

    if magnitude is None:
        raise ValueError(
            "La magnitude de l'événement "
            "USGS est indisponible."
        )

    return {
        "event_id": event_id,
        "mag": float(magnitude),
        "depth": float(depth),
        "sig": float(
            sig if sig is not None else 0
        ),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "tsunami": int(
            tsunami if tsunami is not None else 0
        )
    }


# ============================================================
# SESSION STATE
# ============================================================

if "latitude" not in st.session_state:

    st.session_state.latitude = 45.5017


if "longitude" not in st.session_state:

    st.session_state.longitude = -73.5673


if "damage_prediction" not in st.session_state:

    st.session_state.damage_prediction = None


if "economic_prediction" not in st.session_state:

    st.session_state.economic_prediction = None


if "economic_event" not in st.session_state:

    st.session_state.economic_event = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "USGS Earthquake AI"
)

model_choice = st.sidebar.radio(
    "Choisir un modèle",
    [
        "Prédiction des dommages",
        "Prédiction des pertes économiques"
    ]
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Projet de Machine Learning basé sur "
    "les données sismiques de l'USGS."
)


# ============================================================
# MODEL 5 — DAMAGE PREDICTION
# ============================================================

if model_choice == "Prédiction des dommages":

    st.title(
        "USGS Earthquake AI — "
        "Prédiction des dommages"
    )

    st.write(
        "Sélectionnez un emplacement sur la carte, "
        "entrez les caractéristiques du séisme, "
        "puis lancez la prédiction."
    )

    # ========================================================
    # DATE
    # ========================================================

    st.subheader(
        "1. Choisir une date"
    )

    selected_date = st.date_input(
        "Date",
        value=date.today()
    )

    # ========================================================
    # MAP
    # ========================================================

    st.subheader(
        "2. Choisir un emplacement"
    )

    st.write(
        "Cliquez sur la carte pour sélectionner "
        "l'emplacement du séisme."
    )

    selection_map = folium.Map(
        location=[
            st.session_state.latitude,
            st.session_state.longitude
        ],
        zoom_start=3
    )

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
        tooltip="Emplacement sélectionné"
    ).add_to(selection_map)

    map_data = st_folium(
        selection_map,
        width=None,
        height=500,
        key="damage_selection_map",
        returned_objects=[
            "last_clicked"
        ]
    )

    if (
        map_data
        and map_data.get("last_clicked")
    ):

        clicked_latitude = (
            map_data["last_clicked"]["lat"]
        )

        clicked_longitude = (
            map_data["last_clicked"]["lng"]
        )

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

            st.session_state.latitude = (
                clicked_latitude
            )

            st.session_state.longitude = (
                clicked_longitude
            )

            st.session_state.damage_prediction = None

            st.rerun()

    col_lat, col_lon = st.columns(2)

    col_lat.metric(
        "Latitude",
        round(
            st.session_state.latitude,
            4
        )
    )

    col_lon.metric(
        "Longitude",
        round(
            st.session_state.longitude,
            4
        )
    )

    # ========================================================
    # DAMAGE INPUTS
    # ========================================================

    st.subheader(
        "3. Caractéristiques du séisme"
    )

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
            "Profondeur (km)",
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

    with col2:

        cdi = st.number_input(
            "CDI — Intensité ressentie",
            min_value=0.0,
            max_value=10.0,
            value=4.1,
            step=0.1
        )

        felt = st.number_input(
            "Nombre de signalements ressentis",
            min_value=0,
            value=10,
            step=1
        )

        sig = st.number_input(
            "Indice de signification USGS (SIG)",
            min_value=0,
            value=400,
            step=10
        )

    with col3:

        tsunami_option = st.selectbox(
            "Tsunami",
            [
                "Non",
                "Oui"
            ]
        )

        tsunami = (
            1
            if tsunami_option == "Oui"
            else 0
        )

        st.write(
            "Emplacement"
        )

        st.write(
            f"Latitude : "
            f"{st.session_state.latitude:.4f}"
        )

        st.write(
            f"Longitude : "
            f"{st.session_state.longitude:.4f}"
        )

    # ========================================================
    # DAMAGE PREDICTION
    # ========================================================

    st.subheader(
        "4. Prédiction"
    )

    if st.button(
        "Prédire les dommages",
        type="primary"
    ):

        felt_log_imputed = np.log1p(
            felt
        )

        damage_features = pd.DataFrame(
            [
                {
                    "mag": mag,
                    "depth": depth,
                    "mmi": mmi,
                    "cdi_imputed": cdi,
                    "felt_log_imputed":
                        felt_log_imputed,
                    "sig": sig,
                    "tsunami": tsunami,
                    "latitude":
                        st.session_state.latitude,
                    "longitude":
                        st.session_state.longitude
                }
            ]
        )

        prediction = damage_model.predict(
            damage_features
        )[0]

        st.session_state.damage_prediction = (
            prediction
        )

    # ========================================================
    # DAMAGE RESULT
    # ========================================================

    if (
        st.session_state.damage_prediction
        is not None
    ):

        prediction = (
            st.session_state.damage_prediction
        )

        st.write(
            f"### Résultat pour le {selected_date}"
        )

        if prediction == "Faible":

            st.success(
                "Dommages prédits : FAIBLES"
            )

            marker_color = "green"

        elif prediction == "Modéré":

            st.warning(
                "Dommages prédits : MODÉRÉS"
            )

            marker_color = "orange"

        else:

            st.error(
                "Dommages prédits : SÉVÈRES"
            )

            marker_color = "red"

        result_map = folium.Map(
            location=[
                st.session_state.latitude,
                st.session_state.longitude
            ],
            zoom_start=6
        )

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
            tooltip=(
                f"Dommages prédits : "
                f"{prediction}"
            ),
            popup=folium.Popup(
                f"""
                <b>Date :</b> {selected_date}<br>
                <b>Magnitude :</b> {mag}<br>
                <b>Profondeur :</b> {depth} km<br>
                <b>MMI :</b> {mmi}<br>
                <b>CDI :</b> {cdi}<br>
                <b>SIG :</b> {sig}<br>
                <b>Dommages :</b> {prediction}
                """,
                max_width=350
            )
        ).add_to(result_map)

        st_folium(
            result_map,
            width=None,
            height=500,
            key="damage_result_map",
            returned_objects=[]
        )


# ============================================================
# MODEL 7 — ECONOMIC LOSS
# ============================================================

else:

    st.title(
        "USGS Earthquake AI — "
        "Prédiction des pertes économiques"
    )

    st.write(
        "Entrez l'identifiant d'un séisme USGS. "
        "L'application récupère les caractéristiques "
        "du séisme nécessaires au modèle Model 7."
    )

    st.info(
        "Model 7 utilise un modèle Gradient Boosting "
        "entraîné à partir des caractéristiques "
        "sismiques disponibles."
    )

    # ========================================================
    # EVENT ID
    # ========================================================

    st.subheader(
        "1. Séisme USGS"
    )

    event_id = st.text_input(
        "USGS Event ID",
        value="us7000pn9s",
        placeholder="Exemple : us7000pn9s"
    )

    st.caption(
        "Exemple de test : us7000pn9s"
    )

    # ========================================================
    # ANALYSIS
    # ========================================================

    if st.button(
        "Analyser le séisme",
        type="primary"
    ):

        try:

            with st.spinner(
                "Chargement des données USGS..."
            ):

                event_data = load_usgs_event(
                    event_id
                )

            # ------------------------------------------------
            # BUILD MODEL 7 INPUT
            # ------------------------------------------------

            economic_features = pd.DataFrame(
                [
                    {
                        "mag":
                            event_data["mag"],

                        "depth":
                            event_data["depth"],

                        "sig":
                            event_data["sig"],

                        "latitude":
                            event_data["latitude"],

                        "longitude":
                            event_data["longitude"],

                        "tsunami":
                            event_data["tsunami"]
                    }
                ]
            )

            # ------------------------------------------------
            # USE THE EXACT FEATURES STORED IN THE ARTIFACT
            # ------------------------------------------------

            missing_features = [
                feature
                for feature
                in economic_loss_features
                if feature
                not in economic_features.columns
            ]

            if missing_features:

                raise ValueError(
                    "Variables requises par le modèle "
                    f"absentes : {missing_features}"
                )

            economic_features = (
                economic_features[
                    economic_loss_features
                ]
            )

            # ------------------------------------------------
            # MODEL 7 PREDICTION
            # ------------------------------------------------

            prediction_transformed = (
                economic_loss_model.predict(
                    economic_features
                )[0]
            )

            # ------------------------------------------------
            # RETURN TO DOLLAR SCALE
            # ------------------------------------------------

            economic_prediction = (
                inverse_economic_loss(
                    prediction_transformed
                )
            )

            economic_prediction = max(
                0.0,
                economic_prediction
            )

            st.session_state.economic_prediction = (
                economic_prediction
            )

            st.session_state.economic_event = (
                event_data
            )

        except Exception as error:

            st.session_state.economic_prediction = None

            st.session_state.economic_event = None

            st.error(
                "Impossible de charger ou "
                "d'analyser le séisme USGS : "
                f"{error}"
            )

    # ========================================================
    # ECONOMIC RESULT
    # ========================================================

    if (
        st.session_state.economic_prediction
        is not None
        and
        st.session_state.economic_event
        is not None
    ):

        event_data = (
            st.session_state.economic_event
        )

        economic_prediction = (
            st.session_state.economic_prediction
        )

        st.success(
            "Données USGS chargées avec succès."
        )

        # ====================================================
        # EARTHQUAKE INFORMATION
        # ====================================================

        st.subheader(
            "2. Informations du séisme"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Magnitude",
            event_data["mag"]
        )

        col2.metric(
            "Profondeur",
            f'{event_data["depth"]:.1f} km'
        )

        col3.metric(
            "SIG",
            int(event_data["sig"])
        )

        st.write(
            f'Latitude : '
            f'{event_data["latitude"]:.4f}'
        )

        st.write(
            f'Longitude : '
            f'{event_data["longitude"]:.4f}'
        )

        st.write(
            f'Tsunami : '
            f'{"Oui" if event_data["tsunami"] == 1 else "Non"}'
        )

        # ====================================================
        # MODEL RESULT
        # ====================================================

        st.subheader(
            "3. Résultat du modèle"
        )

        st.metric(
            "Pertes économiques estimées",
            f"{economic_prediction:,.2f} $ US"
        )

        st.caption(
            "Estimation produite par le modèle "
            "Gradient Boosting de Model 7. "
            "Il s'agit d'une estimation de Machine Learning "
            "et non d'une perte économique réelle confirmée."
        )

        # ====================================================
        # MAP
        # ====================================================

        st.subheader(
            "4. Localisation du séisme"
        )

        economic_map = folium.Map(
            location=[
                event_data["latitude"],
                event_data["longitude"]
            ],
            zoom_start=6
        )

        folium.CircleMarker(
            location=[
                event_data["latitude"],
                event_data["longitude"]
            ],
            radius=15,
            color="purple",
            fill=True,
            fill_color="purple",
            fill_opacity=0.8,
            tooltip=(
                f"Pertes estimées : "
                f"{economic_prediction:,.2f} $ US"
            ),
            popup=folium.Popup(
                f"""
                <b>USGS Event ID :</b>
                {event_data["event_id"]}<br>

                <b>Magnitude :</b>
                {event_data["mag"]}<br>

                <b>Profondeur :</b>
                {event_data["depth"]:.1f} km<br>

                <b>SIG :</b>
                {int(event_data["sig"])}<br>

                <b>Tsunami :</b>
                {
                    "Oui"
                    if event_data["tsunami"] == 1
                    else "Non"
                }<br>

                <b>Pertes estimées :</b>
                {economic_prediction:,.2f} $ US
                """,
                max_width=350
            )
        ).add_to(economic_map)

        st_folium(
            economic_map,
            width=None,
            height=500,
            key="economic_result_map",
            returned_objects=[]
        )
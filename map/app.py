import sys
from pathlib import Path






# ============================================================
# PROJECT ROOT
# ============================================================

project_root = Path(__file__).resolve().parent.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# ============================================================
# IMPORTS
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
import requests
from models.action_engine import action_engine
from datetime import date
from streamlit_folium import st_folium
import sklearn.impute


# ============================================================
# PAGE CONFIGURATION
# ============================================================


# Patch de compatibilité pour les modèles entraînés avec une ancienne version de scikit-learn
if not hasattr(sklearn.impute.SimpleImputer, "_fill_dtype"):
    sklearn.impute.SimpleImputer._fill_dtype = property(
        lambda self: getattr(self, "statistics_", np.array([])).dtype
    )


st.set_page_config(
    page_title="USGS Earthquake AI",
    layout="wide"
)


# ============================================================
# LOAD MODELS & ARTIFACTS
# ============================================================

# Model 5 — Damage Model
damage_model = joblib.load(
    "models/damage_random_forest.joblib"
)

# Model 7 — Economic Loss Model
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

# Model 8 — Risk Zone Model
try:
    risk_zone_model = joblib.load(
        "models/Risk_Zone_Random_Forest_champion_model.sekops"
    )
except Exception:
    # Fallback si le fichier a été renommé en .joblib ou .pkl
    risk_zone_model = joblib.load(
        "models/Risk_Zone_Random_Forest_champion_model.joblib"
    )


# ============================================================
# ECONOMIC LOSS INVERSE TRANSFORMATION
# ============================================================

def inverse_economic_loss(value):
    """
    Convert the Model 7 prediction back to the original economic-loss scale.
    """
    if callable(economic_loss_inverse_transform):
        result = economic_loss_inverse_transform(value)
        if np.isscalar(result):
            return float(result)
        return float(np.asarray(result).reshape(-1)[0])

    transform_name = str(economic_loss_inverse_transform).lower()

    if "expm1" in transform_name or "log1p" in transform_name:
        return float(np.expm1(value))

    if transform_name in ["none", "identity", "no_transform"]:
        return float(value)

    if "log" in transform_name:
        return float(np.expm1(value))

    return float(value)


# ============================================================
# LOAD USGS EVENT
# ============================================================

def load_usgs_event(event_id):
    event_id = event_id.strip()
    if not event_id:
        raise ValueError("Veuillez entrer un identifiant USGS.")

    url = (
        "https://earthquake.usgs.gov/fdsnws/event/1/query"
        f"?eventid={event_id}&format=geojson"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    data = response.json()

    if "properties" not in data or "geometry" not in data:
        raise ValueError("Événement USGS introuvable.")

    properties = data["properties"]
    geometry = data["geometry"]
    coordinates = geometry.get("coordinates", [])

    if len(coordinates) < 3:
        raise ValueError("Les coordonnées de l'événement USGS sont indisponibles.")

    longitude = coordinates[0]
    latitude = coordinates[1]
    depth = coordinates[2]
    magnitude = properties.get("mag")
    sig = properties.get("sig")
    tsunami = properties.get("tsunami")

    if magnitude is None:
        raise ValueError("La magnitude de l'événement USGS est indisponible.")

    return {
        "event_id": event_id,
        "mag": float(magnitude),
        "depth": float(depth),
        "sig": float(sig if sig is not None else 0),
        "latitude": float(latitude),
        "longitude": float(longitude),
        "tsunami": int(tsunami if tsunami is not None else 0)
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

if "action_result" not in st.session_state:
    st.session_state.action_result = None

if "economic_prediction" not in st.session_state:
    st.session_state.economic_prediction = None

if "economic_event" not in st.session_state:
    st.session_state.economic_event = None

if "risk_prediction" not in st.session_state:
    st.session_state.risk_prediction = None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("USGS Earthquake AI")

model_choice = st.sidebar.radio(
    "Choisir un modèle",
    [
        "Prédiction des dommages",
        "Prédiction des pertes économiques",
        "Prédiction des zones de risque"
    ]
)

st.sidebar.markdown("---")
st.sidebar.write("Projet de Machine Learning basé sur les données sismiques de l'USGS.")


# ============================================================
# MODEL 5 — DAMAGE PREDICTION
# ============================================================

if model_choice == "Prédiction des dommages":

    st.title("USGS Earthquake AI — Prédiction des dommages")
    st.write(
        "Sélectionnez un emplacement sur la carte, entrez les caractéristiques "
        "du séisme, puis lancez la prédiction."
    )

    # 1. Date
    st.subheader("1. Choisir une date")
    selected_date = st.date_input("Date", value=date.today())

    # 2. Carte
    st.subheader("2. Choisir un emplacement")
    st.write("Cliquez sur la carte pour sélectionner l'emplacement du séisme.")

    selection_map = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=3
    )

    folium.CircleMarker(
        location=[st.session_state.latitude, st.session_state.longitude],
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
        returned_objects=["last_clicked"]
    )

    if map_data and map_data.get("last_clicked"):
        clicked_latitude = map_data["last_clicked"]["lat"]
        clicked_longitude = map_data["last_clicked"]["lng"]

        location_changed = (
            abs(clicked_latitude - st.session_state.latitude) > 0.0001
            or abs(clicked_longitude - st.session_state.longitude) > 0.0001
        )

        if location_changed:
            st.session_state.latitude = clicked_latitude
            st.session_state.longitude = clicked_longitude
            st.session_state.damage_prediction = None
            st.session_state.action_result = None
            st.rerun()

    col_lat, col_lon = st.columns(2)
    col_lat.metric("Latitude", round(st.session_state.latitude, 4))
    col_lon.metric("Longitude", round(st.session_state.longitude, 4))

    # 3. Formulaire d'entrées
    st.subheader("3. Caractéristiques du séisme")
    col1, col2, col3 = st.columns(3)

    with col1:
        mag = st.number_input("Magnitude", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        depth = st.number_input("Profondeur (km)", min_value=0.0, max_value=800.0, value=10.0, step=1.0)
        mmi = st.number_input("MMI", min_value=0.0, max_value=12.0, value=5.0, step=0.1)

    with col2:
        cdi = st.number_input("CDI — Intensité ressentie", min_value=0.0, max_value=10.0, value=4.1, step=0.1)
        felt = st.number_input("Nombre de signalements ressentis", min_value=0, value=10, step=1)
        sig = st.number_input("Indice de signification USGS (SIG)", min_value=0, value=400, step=10)

    with col3:
        tsunami_option = st.selectbox("Tsunami", ["Non", "Oui"])
        tsunami = 1 if tsunami_option == "Oui" else 0
        st.write("Emplacement")
        st.write(f"Latitude : {st.session_state.latitude:.4f}")
        st.write(f"Longitude : {st.session_state.longitude:.4f}")

    # 4. Prédiction
    st.subheader("4. Prédiction")

    if st.button("Prédire les dommages", type="primary"):
        felt_log_imputed = np.log1p(felt)
        damage_features = pd.DataFrame([{
            "mag": mag,
            "depth": depth,
            "mmi": mmi,
            "cdi_imputed": cdi,
            "felt_log_imputed": felt_log_imputed,
            "sig": sig,
            "tsunami": tsunami,
            "latitude": st.session_state.latitude,
            "longitude": st.session_state.longitude
        }])

        prediction = damage_model.predict(damage_features)[0]
        st.session_state.damage_prediction = prediction
        st.session_state.action_result = None

    # Affichage du résultat
    if st.session_state.damage_prediction is not None:
        prediction = st.session_state.damage_prediction
        st.write(f"### Résultat pour le {selected_date}")

        if prediction == "Faible":
            st.success("Dommages prédits : FAIBLES")
            marker_color = "green"
        elif prediction == "Modéré":
            st.warning("Dommages prédits : MODÉRÉS")
            marker_color = "orange"
        else:
            st.error("Dommages prédits : SÉVÈRES")
            marker_color = "red"

        result_map = folium.Map(
            location=[st.session_state.latitude, st.session_state.longitude],
            zoom_start=6
        )

        folium.CircleMarker(
            location=[st.session_state.latitude, st.session_state.longitude],
            radius=15,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.8,
            tooltip=f"Dommages prédits : {prediction}",
            popup=folium.Popup(f"""
                <b>Date :</b> {selected_date}<br>
                <b>Magnitude :</b> {mag}<br>
                <b>Profondeur :</b> {depth} km<br>
                <b>MMI :</b> {mmi}<br>
                <b>CDI :</b> {cdi}<br>
                <b>SIG :</b> {sig}<br>
                <b>Dommages :</b> {prediction}
            """, max_width=350)
        ).add_to(result_map)

        st_folium(
            result_map,
            width=None,
            height=500,
            key="damage_result_map",
            returned_objects=[]
        )

        # 5. Model 6 — Action Engine
        action_result = action_engine(
            damage_level=prediction,
            tsunami=tsunami
        )
        st.session_state.action_result = action_result
        action, priority, warning, reason = action_result

        st.subheader("5. Action recommandée")
        col_action, col_priority = st.columns(2)

        with col_action:
            st.metric("Action recommandée", action)
        with col_priority:
            st.metric("Priorité", priority)

        if tsunami == 1:
            st.warning(warning)
        else:
            st.info(warning)

        st.write(f"**Justification :** {reason}")


# ============================================================
# MODEL 8 — RISK ZONE PREDICTION
# ============================================================

elif model_choice == "Prédiction des zones de risque":

    st.title("USGS Earthquake AI — Prédiction des zones de risque")
    st.write(
        "Sélectionnez un point sur la carte ou ajustez les coordonnées "
        "et paramètres du séisme pour évaluer le niveau de risque géographique."
    )

    # 1. Sélection de l'emplacement via carte
    st.subheader("1. Localisation géographique")
    
    risk_map = folium.Map(
        location=[st.session_state.latitude, st.session_state.longitude],
        zoom_start=3
    )

    folium.CircleMarker(
        location=[st.session_state.latitude, st.session_state.longitude],
        radius=8,
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.8,
        tooltip="Zone sélectionnée"
    ).add_to(risk_map)

    map_risk_data = st_folium(
        risk_map,
        width=None,
        height=450,
        key="risk_selection_map",
        returned_objects=["last_clicked"]
    )

    if map_risk_data and map_risk_data.get("last_clicked"):
        clicked_lat = map_risk_data["last_clicked"]["lat"]
        clicked_lon = map_risk_data["last_clicked"]["lng"]

        if (abs(clicked_lat - st.session_state.latitude) > 0.0001 or
            abs(clicked_lon - st.session_state.longitude) > 0.0001):
            st.session_state.latitude = clicked_lat
            st.session_state.longitude = clicked_lon
            st.session_state.risk_prediction = None
            st.rerun()

    # 2. Saisie des 10 variables requises par le modèle
    st.subheader("2. Caractéristiques et paramètres de la zone")

    col_lat, col_lon = st.columns(2)
    with col_lat:
        lat = st.number_input(
            "Latitude", 
            value=float(st.session_state.latitude), 
            format="%.4f"
        )
    with col_lon:
        lon = st.number_input(
            "Longitude", 
            value=float(st.session_state.longitude), 
            format="%.4f"
        )

    # Synchronisation des coordonnées en session
    if lat != st.session_state.latitude or lon != st.session_state.longitude:
        st.session_state.latitude = lat
        st.session_state.longitude = lon

    # Paramètres physiques et mesures sismiques
    col1, col2, col3 = st.columns(3)
    with col1:
        depth = st.number_input("Profondeur — depth (km)", min_value=0.0, max_value=800.0, value=10.0, step=1.0)
        dmin = st.number_input("Distance min. station — dmin", min_value=0.0, value=0.5, step=0.1)

    with col2:
        nst = st.number_input("Nombre de stations — nst", min_value=0, value=20, step=1)
        gap = st.number_input("Écart azimutal — gap (°)", min_value=0.0, max_value=360.0, value=90.0, step=1.0)

    with col3:
        rms = st.number_input("Erreur RMS — rms", min_value=0.0, value=0.5, step=0.01)
        net = st.selectbox("Réseau sismique — net", ["us", "ci", "nc", "uw", "ak", "nn", "pr", "uu"])

    col4, col5 = st.columns(2)
    with col4:
        magType = st.selectbox("Type de magnitude — magType", ["mb", "mww", "mwr", "ml", "md", "mb_lg"])
    with col5:
        event_type = st.selectbox("Type d'événement — type", ["earthquake", "quarry blast", "explosion", "ice quake"])

    # 3. Exécution de la prédiction
    st.subheader("3. Analyse de la zone")
    if st.button("Vérifier la zone de risque", type="primary"):
        # Construction du DataFrame exact avec les 10 colonnes requises
        risk_features = pd.DataFrame([{
            "latitude": lat,
            "longitude": lon,
            "depth": depth,
            "dmin": dmin,
            "nst": nst,
            "gap": gap,
            "rms": rms,
            "net": net,
            "magType": magType,
            "type": event_type
        }])

        try:
            prediction = risk_zone_model.predict(risk_features)[0]
            st.session_state.risk_prediction = prediction
        except Exception as e:
            st.error(f"Erreur lors de l'exécution du modèle : {e}")

    # 4. Affichage du résultat
    if st.session_state.risk_prediction is not None:
        pred = st.session_state.risk_prediction
        st.subheader("Résultat de l'évaluation")

        if pred in [2, "Élevé", "High", "Risque Élevé"]:
            st.error(f"⚠️ Zone identifiée à HAUT RISQUE sismique ({pred})")
        elif pred in [1,"Modéré", "Medium", "Risque Modéré"]:
            st.warning(f"⚡ Zone identifiée à RISQUE MODÉRÉ ({pred})")
        else:
            st.success(f"✅ Zone identifiée à FAIBLE RISQUE ({pred})")

# ============================================================
# MODEL 7 — ECONOMIC LOSS
# ============================================================

else:

    st.title("USGS Earthquake AI — Prédiction des pertes économiques")
    st.write(
        "Entrez l'identifiant d'un séisme USGS. L'application récupère "
        "les caractéristiques du séisme nécessaires au modèle Model 7."
    )
    st.info("Model 7 utilise un modèle Gradient Boosting entraîné à partir des caractéristiques sismiques disponibles.")

    st.subheader("1. Séisme USGS")
    event_id = st.text_input("USGS Event ID", value="us7000pn9s", placeholder="Exemple : us7000pn9s")
    st.caption("Exemple de test : us7000pn9s")

    if st.button("Analyser le séisme", type="primary"):
        try:
            with st.spinner("Chargement des données USGS..."):
                event_data = load_usgs_event(event_id)

            economic_features = pd.DataFrame([{
                "mag": event_data["mag"],
                "depth": event_data["depth"],
                "sig": event_data["sig"],
                "latitude": event_data["latitude"],
                "longitude": event_data["longitude"],
                "tsunami": event_data["tsunami"]
            }])

            missing_features = [
                feature for feature in economic_loss_features
                if feature not in economic_features.columns
            ]

            if missing_features:
                raise ValueError(f"Variables requises par le modèle absentes : {missing_features}")

            economic_features = economic_features[economic_loss_features]

            prediction_transformed = economic_loss_model.predict(economic_features)[0]
            economic_prediction = inverse_economic_loss(prediction_transformed)
            economic_prediction = max(0.0, economic_prediction)

            st.session_state.economic_prediction = economic_prediction
            st.session_state.economic_event = event_data

        except Exception as error:
            st.session_state.economic_prediction = None
            st.session_state.economic_event = None
            st.error(f"Impossible de charger ou d'analyser le séisme USGS : {error}")

    if st.session_state.economic_prediction is not None and st.session_state.economic_event is not None:
        event_data = st.session_state.economic_event
        economic_prediction = st.session_state.economic_prediction

        st.success("Données USGS chargées avec succès.")

        st.subheader("2. Informations du séisme")
        col1, col2, col3 = st.columns(3)
        col1.metric("Magnitude", event_data["mag"])
        col2.metric("Profondeur", f'{event_data["depth"]:.1f} km')
        col3.metric("SIG", int(event_data["sig"]))

        st.write(f'Latitude : {event_data["latitude"]:.4f}')
        st.write(f'Longitude : {event_data["longitude"]:.4f}')
        st.write(f'Tsunami : {"Oui" if event_data["tsunami"] == 1 else "Non"}')

        st.subheader("3. Résultat du modèle")
        st.metric("Pertes économiques estimées", f"{economic_prediction:,.2f} $ US")
        st.caption(
            "Estimation produite par le modèle Gradient Boosting de Model 7. "
            "Il s'agit d'une estimation de Machine Learning et non d'une perte économique réelle confirmée."
        )

        st.subheader("4. Localisation du séisme")
        economic_map = folium.Map(
            location=[event_data["latitude"], event_data["longitude"]],
            zoom_start=6
        )

        folium.CircleMarker(
            location=[event_data["latitude"], event_data["longitude"]],
            radius=15,
            color="purple",
            fill=True,
            fill_color="purple",
            fill_opacity=0.8,
            tooltip=f"Pertes estimées : {economic_prediction:,.2f} $ US",
            popup=folium.Popup(f"""
                <b>USGS Event ID :</b> {event_data["event_id"]}<br>
                <b>Magnitude :</b> {event_data["mag"]}<br>
                <b>Profondeur :</b> {event_data["depth"]:.1f} km<br>
                <b>SIG :</b> {int(event_data["sig"])}<br>
                <b>Tsunami :</b> {"Oui" if event_data["tsunami"] == 1 else "Non"}<br>
                <b>Pertes estimées :</b> {economic_prediction:,.2f} $ US
            """, max_width=350)
        ).add_to(economic_map)

        st_folium(
            economic_map,
            width=None,
            height=500,
            key="economic_result_map",
            returned_objects=[]
        )
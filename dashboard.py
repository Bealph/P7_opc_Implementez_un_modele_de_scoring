import streamlit as st
import pandas as pd
from PIL import Image
import json
import requests

#--------------------------------------------
# Charger l'image
image = Image.open('image_app.jpeg')

# Afficher une colonne pour centrer l'image
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(image, width=250, output_format="JPEG")

# ----------------------------
# le body
    
st.markdown(
    """
<style>
.header {
    font-size : 24px;
    text-align : center;
    text-decoration : underline;
}
</style>

<style>
.centered-text {
    text-align : justify;
    text-align-last : center;
}
</style>

<style>
.footer {
    position : fixed;
    bottom : 0;
    left : 0;
    width : 100%;
    background-color : white;
    text-align : justify;
    text-align-last : center;
    color : blue;
    font-size : 8px;
    
    p {
        margin-bottom: 0px;
    }
}
</style>

""",
unsafe_allow_html=True
)


st.markdown(
    '<h1 class="header">Tableau de bord en temps réel</h1>',
    unsafe_allow_html=True
)

# costomisation texte d'explication de l'application
st.markdown(
    '<p class="centered-text">Cette application vise à fournir un système de scoring avancé permettant\
        l\'accès aux informations essentielles des clients. Grâce à des modèles prédictifs, elle offre la possibilité\
            de générer des predictions pertinentes en temps réel.<br> <br> L\'objectif principale est de faciliter \
                la prise de décision en fournissant des analysées détaillées basées sur les données des clients.</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer"> <p>2023 DIALLO Alpha</p>\
        <p>P7 OPC</p>\
        </div>', unsafe_allow_html=True
)

# --------------------------------------------

# Charger les données depuis votre fichier CSV
client_data = pd.read_csv("top_50_train.csv", encoding='utf-8')

# Menu déroulant pour sélectionner un client
selected_client = st.sidebar.selectbox("Sélectionnez un client", client_data['SK_ID_CURR'])

# Variables de contrôle pour afficher ou non les éléments
show_variables = False
show_predictions = False
show_close_button = False

# Si un client est sélectionné, afficher les boutons correspondants
if selected_client:
    # Afficher les boutons et récupérer leur état
    show_variables = st.sidebar.button("Afficher les 10 variables importantes", key="variables_button")
    show_predictions = st.sidebar.button("Affichage des prédictions probables", key="predictions_button")

    # Gestion de l'affichage des boutons
    if show_variables:
        show_close_button = True

    if show_predictions:
        show_close_button = True



# Affichage des éléments dans la colonne principale en fonction des boutons cliqués
if show_variables:
    # Cacher les autres éléments dans la colonne principale
    st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)
     # Obtenir les données du client sélectionné
    # Au lieu de client_data = df[df['SK_ID_CURR'] == selected_client].iloc[0].to_dict()
    client_data_dict = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0].to_dict()

    # Convertir le dictionnaire en JSON
    client_data_json = json.dumps(client_data_dict)

    # Faire une requête à votre API Flask
    url = "http://127.0.0.1:5000/api/predict_proba"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=client_data_json)
        if response.status_code == 200:
            result = response.json()
            # Assurez-vous que 'important_variables' correspond aux noms des 10 variables les plus importantes
            important_variables = result['important_variables']
            st.write(pd.DataFrame(important_variables))  # Afficher le tableau des 10 variables

        else:
            st.error(f"Échec de la requête : {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la requête : {e}")

#-------------                --------------------

if show_predictions:
    # Cacher les autres éléments dans la colonne principale
    st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)

    #
    client_data_dict = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0].to_dict()

    # Convertir le dictionnaire en JSON
    client_data_json = json.dumps(client_data_dict)

    # Faire une requête à ton API Flask pour récupérer les prédictions
    url = "http://127.0.0.1:5000/api/predict_proba"
    headers = {"Content-Type": "application/json"}

    try:
        st.write(f"Data sent to Flask API: {client_data}")  # Affiche les données envoyées à l'API Flask
        response = requests.post(url, headers=headers, json=client_data_json)
        st.write(f"API Response: {response}")  # Affiche la réponse de l'API
        st.write(f"API Response JSON: {response.json()}")  # Affiche le JSON de la réponse de l'API

        if response.status_code == 200:
            result = response.json()
            prediction_proba = result['proba']
            st.write(f"Prédictions probables : {prediction_proba}")  # Afficher les prédictions probables

        else:
            st.error(f"Échec de la requête : {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la requête : {e}")


# Afficher le bouton "Fermer l'affichage" s'il y a quelque chose à fermer
if show_close_button:
    if st.sidebar.button("Fermer l'affichage"):
        # Réinitialisation de l'affichage à l'étape initiale
        st.sidebar.markdown('<style>#variables_button, #predictions_button {display: block;}</style>', unsafe_allow_html=True)
        st.markdown('<style>.header, .centered-text, img {display: block;}</style>', unsafe_allow_html=True)
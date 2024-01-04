import streamlit as st
import pandas as pd
from PIL import Image
import brouillon_2 as br2  # Le fichier pour l'envoi des requêtes

# Charger l'image
image = Image.open('image_app.jpeg')

# Afficher une colonne pour centrer l'image
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.image(image, width=250, output_format="JPEG")

# Le corps de l'application
st.markdown(
    """
<style>
.header {
    font-size : 24px;
    text-align : center;
    text-decoration : underline;
}
.centered-text {
    text-align : justify;
    text-align-last : center;
}
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

st.markdown('<h1 class="header">Tableau de bord en temps réel</h1>', unsafe_allow_html=True)

st.markdown(
    '<p class="centered-text">Cette application vise à fournir un système de scoring avancé permettant\
    l\'accès aux informations essentielles des clients. Grâce à des modèles prédictifs, elle offre la possibilité\
    de générer des prédictions pertinentes en temps réel.<br> <br> L\'objectif principal est de faciliter \
    la prise de décision en fournissant des analyses détaillées basées sur les données des clients.</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="footer"> <p>2023 DIALLO Alpha</p>\
        <p>P7 OPC</p>\
        </div>', unsafe_allow_html=True
)

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

    prediction_proba, feature_importance = br2.get_infos_clients(client_data)  # Récupérer les informations pour tous les clients
    

    # Affichage des informations dans un tableau
    if feature_importance:
        st.write("Informations des clients :")
        st.table(feature_importance)
    else:
        st.write("Aucune information disponible pour les clients.")

# Affichage des éléments dans la colonne principale en fonction des boutons cliqués
if show_variables:
    # Cacher les autres éléments dans la colonne principale
    st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)

    prediction_proba, feature_importance = br2.get_infos_clients(client_data) # Récupérer les informations pour tous les clients

    # Si 'all_clients_information' contient les données sous forme de dictionnaire
    # Supposons que les valeurs importantes sont stockées sous une clé 'important_values'
    if feature_importance and 'important_values' in feature_importance:
        important_values = feature_importance['important_values']
        
        # Trier les variables par ordre de grandeur et sélectionner les 10 premières
        sorted_values = sorted(important_values.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Affichage des 10 variables les plus importantes dans un tableau
        st.write("Les 10 variables les plus importantes :")
        st.write(sorted_values)
    else:
        st.write("Aucune information trouvée pour les clients.")



# Afficher le 4 top des variable un boxplot et un point rouge pour voir où se trouve le client, un histogramme

# -------------                --------------------
# Autres sections et fonctionnalités
# ...

# Afficher le bouton "Fermer l'affichage" s'il y a quelque chose à fermer
if show_close_button:
    if st.sidebar.button("Fermer l'affichage"):
        # Réinitialisation de l'affichage à l'étape initiale
        st.sidebar.markdown('<style>#variables_button, #predictions_button {display: block;}</style>', unsafe_allow_html=True)
        st.markdown('<style>.header, .centered-text, img {display: block;}</style>', unsafe_allow_html=True)

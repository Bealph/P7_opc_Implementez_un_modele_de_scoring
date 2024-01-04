import streamlit as st
import pandas as pd
from PIL import Image
import request_app as ra

#---------------------------------
# pour les graphique shap, les importer les shapvalues(en les important au format pickle), pour pouvoir faire facilementle graphique

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

    client_data = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0].drop(labels='SK_ID_CURR')

    prediction_proba, feature_names, feature_importance = ra.get_infos_client(client_data)

    top_10_indices = sorted(range(len(feature_importance)), key=lambda i: feature_importance[i], reverse=True)[:10]
    top_10_features = [(feature_names[i], feature_importance[i]) for i in top_10_indices]
    top_10_df = pd.DataFrame(top_10_features, columns=['Variables', 'Importances'])

    st.write("Les 10 variables les plus importantes :")
    st.table(top_10_df)




# Afficher le 4 top des variable un boxplot et un point rouge pour voir où se trouve le client, un histogramme




#-------------                --------------------

if show_predictions:
    # Cacher les autres éléments dans la colonne principale
    st.markdown('<style>.header, .centered-text, img {display: none;}</style>', unsafe_allow_html=True)

    #
    client_data = client_data[client_data['SK_ID_CURR'] == selected_client].iloc[0].drop(labels='SK_ID_CURR')

    prediction_proba, feature_names, feature_importance = ra.get_infos_client(client_data)
    st.write(prediction_proba)

    # afficher un decision plot, ou un donut plot, un chartplot
    






# Afficher le bouton "Fermer l'affichage" s'il y a quelque chose à fermer
if show_close_button:
    if st.sidebar.button("Fermer l'affichage"):
        # Réinitialisation de l'affichage à l'étape initiale
        st.sidebar.markdown('<style>#variables_button, #predictions_button {display: block;}</style>', unsafe_allow_html=True)
        st.markdown('<style>.header, .centered-text, img {display: block;}</style>', unsafe_allow_html=True)
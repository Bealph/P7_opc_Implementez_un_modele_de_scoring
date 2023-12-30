import streamlit as st
import pandas as pd
import requests

st.markdown(
    """
<style>
.centered {
    display : flex;
    justify-content : center;
}
</style>

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
    '<div class="centered"><img src="/img_test.PNG"></div>',
    unsafe_allow_html=True
)

# le header
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

# Charger les données depuis votre fichier CSV
df = pd.read_csv("top_50_train.csv", encoding='utf-8')

# Créer le menu déroulant avec les IDs des clients dans la sidebar
selected_client = st.sidebar.selectbox("Sélectionnez un client", df['SK_ID_CURR'])

# Variables de contrôle pour afficher ou non le tableau des variables importantes
show_variables = False

# Bouton pour afficher les 10 variables importantes dans la sidebar
if st.sidebar.button("Afficher les 10 variables importantes"):
    show_variables = True
    # Hide other elements in the main body
    st.markdown(
        '<style>.header, .centered-text, img {display: none;}</style>', 
        unsafe_allow_html=True
    )
    # Obtenir les données du client sélectionné
    client_data = df[df['SK_ID_CURR'] == selected_client].iloc[0].to_dict()

    # Faire une requête à votre API Flask
    url = "http://127.0.0.1:5000/api/predict_proba"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=client_data)
        if response.status_code == 200:
            result = response.json()
            # Assurez-vous que 'important_variables' correspond aux noms des 10 variables les plus importantes
            important_variables = result['important_variables']
            st.write(pd.DataFrame(important_variables))  # Afficher le tableau des 10 variables

        else:
            st.error(f"Échec de la requête : {response.status_code}")

    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la requête : {e}")

# Bouton pour fermer l'affichage du tableau des variables importantes
if show_variables and st.sidebar.button("Fermer l'affichage"):
    show_variables = False
    # Réafficher les autres éléments dans la colonne body
    st.markdown(
        '<style>.header, .centered-text, img {display: block;}</style>', 
        unsafe_allow_html=True
    )
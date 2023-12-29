import streamlit as st
import pandas as pd

menu_selector = st.sidebar.selectbox('Prediction probable sur le client',
                                     ['proba'])
col1 = st.columns

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

#st.header('Tableau de bord en temps réel')
st.markdown(
    '<h1 class="header">Tableau de bord en temps réel</h1>',
    unsafe_allow_html=True
)

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
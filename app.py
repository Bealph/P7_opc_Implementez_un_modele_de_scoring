# -*- coding: utf-8 -*-

from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
import streamlit as st
from threading import Thread

# Mettre à jour scikit-learn
os.system('pip install --upgrade scikit-learn')

# Suppression des warnings
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------

# Chemin vers le fichier pickle contenant le modèle
chemin_fichier_pickle = r"C:\Users\Microsoft\Desktop\Dossiers formation\Formation OC\Data science\P7\Donnees projet\P7_opc_Implementez_un_modele_de_scoring\mon_best_modele_entraine_LightGBM.pkl"

with open(chemin_fichier_pickle, 'rb') as fichier:
    print("utilisation du meilleur modele : lightGBM")
    modele = pickle.load(fichier)

# ------------------------------
    
# Flask setup
app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.get_json()
    df_data = pd.DataFrame(data)
    prediction_proba = modele.predict_proba(df_data)
    return jsonify({'prediction_proba': prediction_proba.tolist(), 'client_info': data})


#--------------------------------------------------------
# Streamlit setup
def streamlit_thread():
    def streamlit_code():
        st.title("application Streamlit")

    sthread = Thread(target=streamlit_code)
    sthread.start()

if __name__ == '__main__':
    sthread = Thread(target=streamlit_thread)
    sthread.start()
    app.run(debug=True)

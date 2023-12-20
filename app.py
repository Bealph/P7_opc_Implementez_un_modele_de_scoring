

from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle
import os, sys

# -*- coding: utf-8 -*-

# Mettre à jour scikit-learn
os.system('pip install --upgrade scikit-learn')

# Suppression des warnings
import warnings
warnings.filterwarnings('ignore')

# Chemin vers le fichier pickle contenant le modèle
chemin_fichier_pickle = r"C:\Users\Microsoft\Desktop\Dossiers formation\Formation OC\Data science\P7\Donnees projet\P7_opc_Implementez_un_modele_de_scoring\mon_best_modele_entraine_LightGBM.pkl"

with open(chemin_fichier_pickle, 'rb') as fichier:
    print("utilisation du meilleur modele : lightGBM")
    modele = pickle.load(fichier)

# Ajout au code existant

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    # Récupérer les données envoyées depuis le client (test_app.py)
    data = request.get_json()
    
    # Transformer les données en dataframe
    df_data = pd.DataFrame(data)
    
    # Faire une prédiction de probabilité
    prediction_proba = modele.predict_proba(df_data)
    
    # Retourner le résultat sous forme de JSON
    return jsonify({'prediction_proba': prediction_proba.tolist(), 'client_info': data})

if __name__ == '__main__':
    app.run(debug=True)


# streamlit run app.py
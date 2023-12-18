# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle
import imblearn

app = Flask(__name__)

# Chemin vers le fichier pickle contenant le modèle
chemin_fichier_pickle = r"C:\Users\Microsoft\Desktop\Dossiers formation\Formation OC\Data science\P7\Donnees projet\mon_best_modele_entraine_regLog.pkl"

with open(chemin_fichier_pickle, 'rb') as fichier:
    print("utilisation modele reg_log_GridCV")
    modele = pickle.load(fichier)

# Chargeons les données, df, et num_client ici...
"""df = pd.read_csv('application_train.csv')
df.drop(columns='TARGET', inplace=True)
print(df.head())"""

"""print( )

num_client = df['SK_ID_CURR'].unique()
print(f'Numeros de client :', num_client)"""

@app.route('/')
"""def home():
    return render_template('index.html')"""

@app.route('/predict/')
def predict():
    """

    Returns
    liste des clients dans le fichier

    """
    return jsonify({"model": "'reg_log_GridCV",
                    "list_client_id" : list(predict.proba.astype(str))})

@app.route('/predict/<int:sk_id>')
def predict_get(sk_id):
    """

    Parameters
    ----------
    sk_id : numero de client

    Returns
    -------
    prediction  0 pour paiement OK
                1 pour defaut de paiement

    """ Pas de df ici, il faut recuperer le JSON
    if sk_id in num_client:
        predict = modele.predict(df[df['SK_ID_CURR'] == sk_id])[0]
        predict_proba = modele.predict_proba(df[df['SK_ID_CURR'] == sk_id])[0]
        predict_proba_0 = str(predict_proba[0])
        predict_proba_1 = str(predict_proba[1])
    else:
        predict = predict_proba_0 = predict_proba_1 = "client inconnu"
    return jsonify({'retour_prediction': str(predict), 'predict_proba_0': predict_proba_0,
                    'predict_proba_1': predict_proba_1})

if __name__ == '__main__':
    app.run()


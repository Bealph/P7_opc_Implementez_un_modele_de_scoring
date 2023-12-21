
from flask import Flask, jsonify, request
import pandas as pd
import pickle

# Suppression des warnings
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------

# Chemin vers le fichier pickle contenant le modèle
chemin_fichier_pickle = "mon_best_modele_entraine_LightGBM.pkl"

with open(chemin_fichier_pickle, 'rb') as fichier:
    print("utilisation du meilleur modele : lightGBM")
    modele = pickle.load(fichier)

# ------------------------------
    
if __name__ == '__main__':
    # Flask setup
    app = Flask(__name__)

    @app.route('/prediction', methods=['POST'])
    def prediction():
        data = request.get_json()
        df_data = pd.DataFrame(data)
        prediction_proba = modele.predict_proba(df_data)
        return jsonify({'prediction_proba': prediction_proba.tolist(), 'client_info': data})

    app.run(debug=True, port=8200)


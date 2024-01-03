from flask import Flask, jsonify, request
import pandas as pd
import pickle
import lightgbm

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

# Routes Flask
app = Flask(__name__)
app.config['TESTING'] = True

@app.route('/api/predict_proba/', methods=['POST'])
def prediction():
    # Get input data from the request
    data = request.get_json()

    df_data = pd.DataFrame([data])

    # Make a prediction
    prediction_proba = modele.predict_proba(df_data)

    # Prepare the response
    response = {'proba': prediction_proba[0].tolist()}

    return jsonify(response)

# ----------------------------------------

@app.route('/api/client_data/', methods=['POST'])
def receive_data():
    data = request.get_json()
    
    # Convertir les données JSON en DataFrame
    df_data = pd.DataFrame([data])

    # Récupérer les noms de colonnes et les importances (utilisation factice d'importances ici)
    column_names = df_data.columns.tolist()
    importances = [0.5, 0.2, 0.1, 0.3, 0.4, 0.6, 0.9, 0.8, 0.7, 0.2]  # Remplace par tes vraies importances

    # Créer un dictionnaire avec les noms de colonnes et leurs importances
    column_importance = dict(zip(column_names, importances))

    # Trier les colonnes par importance (du plus grand au plus petit) et sélectionner les 10 premières
    top_10_variables = sorted(column_importance, key=column_importance.get, reverse=True)[:10]

    # Récupérer les valeurs des 10 variables les plus importantes
    top_10_values = df_data[top_10_variables].values.tolist()[0]

    # Préparer la réponse en renvoyant les noms et les valeurs des 10 variables les plus importantes
    response = {'top_variables': {var: val for var, val in zip(top_10_variables, top_10_values)}}

    return jsonify(response)

if __name__ == '__main__':
    # Lancement de l'application Flask
    app.run(debug=True)

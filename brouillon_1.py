from flask import Flask, jsonify, request
import pandas as pd
import pickle
import warnings

app = Flask(__name__)
app.config['TESTING'] = True

# Chemin vers le fichier pickle contenant le modèle
chemin_fichier_pickle = "mon_best_modele_entraine_LightGBM.pkl"

with open(chemin_fichier_pickle, 'rb') as fichier:
    print("utilisation du meilleur modele : lightGBM")
    modele = pickle.load(fichier)

# Routes Flask
@app.route('/api/infos_client/', methods=['POST'])
def data_client():
    # Get input data from the request
    data = request.get_json()

    df_data = pd.DataFrame([data])

    # Make a prediction
    prediction_proba = modele.predict_proba(df_data)
    
    # Get feature importances from the trained model
    feature_importance = modele.feature_importances_.tolist()

    # Prepare the response
    response = {'proba': prediction_proba[0].tolist(), 'vars': feature_importance}

    return jsonify(response)

if __name__ == '__main__':
    # Lancement de l'application Flask
    app.run(debug=True)

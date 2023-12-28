
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
    
if __name__ == '__main__':
    # Flask setup
    app = Flask(__name__)
    app.config['TESTING'] = True

    @app.route('/api/predict_proba/', methods=['POST'])
    def prediction():
        # Get input data from the request
        data = request.get_json()

        df_data = pd.DataFrame([data])

# Make a prediction
        prediction_proba = modele.predict_proba(df_data)

        #print(prediction_proba)

        # Prepare the response
        response = {'proba': prediction_proba[0].tolist()}

        print(response)
        
        return jsonify(response)
        #return jsonify('hello')
      #  return jsonify({'prediction_proba': prediction_proba.tolist(), 'client_info': data})

    if __name__=='__main__':
        app.run(debug=True)#, port=8200)


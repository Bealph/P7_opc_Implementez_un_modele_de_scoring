# -*- coding: utf-8 -*-
from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Chemin vers le fichier pickle contenant le modèle
chemin_fichier_pickle = r"C:\Users\Microsoft\Desktop\Dossiers formation\Formation OC\Data science\P7\Donnees projet\P7_opc_Implementez_un_modele_de_scoring\mon_best_modele_entraine_LightGBM.pkl"

with open(chemin_fichier_pickle, 'rb') as fichier:
    print("utilisation modele lightGBM")
    modele = pickle.load(fichier)

# Chargeons les données, df, et num_client ici...
df = pd.read_csv(r"C:/Users/Microsoft/Desktop/Dossiers formation/Formation OC/Data science/P7/Donnees projet/P7_opc_Implementez_un_modele_de_scoring/top_50_train.csv", encoding='utf-8')

if __name__ == '__main__':
    app.run()

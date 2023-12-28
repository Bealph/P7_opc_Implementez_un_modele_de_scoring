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


'''
redesign du dash (image, texte d'expli, en quoi va concerner le dash, 5 a 6 ligne,  un dropdown(menu : id des clients, qui va permettre qd on clique sur l'id, ça affiche sous forme de tableau (max: 10 variables les plus important)
 les info du client), un bouton submit (qd on clique ça puisse envoyer les proba, les données à l'app, créer une fonction, en utilisant comme parametre les id client), et afficher dans le dashboard la liste des proba predict)


'''
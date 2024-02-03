from setuptools import setup, find_packages

setup(
    name='NomDeVotreApplication',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        # Liste des dépendances de votre application
        'Flask==3.0.0',
        'autre_bibliotheque==1.2.3',
        # Ajoutez toutes vos dépendances avec leurs versions ici
    ],
    entry_points={
        'console_scripts': [
            'votre_commande = nom_de_votre_module:fonction_principale',
            # Ajoutez ici d'éventuelles commandes console associées à votre application
        ],
    },
    classifiers=[
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)


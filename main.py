from my_module.tree_classifier import TreeClassifier
from my_module.data_visualizer import seaborn_split, visualize_initial_data
import numpy as np
import pandas as pd
import random

def load_data(file_path):
    """
    Charge et lit proprement le fichier CSV.
    
    Args:
        file_path: chemin vers le fichier à lire

    Returns:
        la dataframe contenant la donnée lue
    """

    try:
        df = pd.read_csv(file_path)

    except FileNotFoundError:
        print("Fichier non trouvé !.")

    except pd.errors.EmptyDataError:
        print("Fichier vide !.")

    except pd.errors.ParserError as e:
        print(f"Erreur lors de la lecture du CSV: {e}")

    except Exception as e:
        print(f"Erreur inattendue: {e}")

    return df

def check_data_validity(df):
    """
    Prend l'objet réprésentant la donnée en paramètre, 
    vérifie qu'il n'y a pas de valeurs nulles et supprime 
    les colonnes inutiles.

    Args:
        df (object): la donnée à traiter

    Returns:
        df (object): la donnée traitée
    """
    # print("\ndf.describe() => \n")
    # print(df.describe())

    # print("\ndf.info() => \n")
    # df.info()

    # supprime la colonne 'Id' qui ne sera pas utile ici
    df = df.drop('Id', axis=1)

    # print("\ndf.head(10) => \n")
    # print(df.head(10))
    return df

def train_test_split(df, test_size, random_state=None):
    """
    Prend une dataframe en paramètre, le pourcentage (ou le nombre de lignes) 
    de la dataframe à dédier au test et retourne respectivement la donnée 
    à entraîner (train_data) et la donnée à tester (test_data).

    Args:
        df (pandas.DataFrame): la dataframe contenant la donnée à séparer
        test_size (float): le pourcentage (entre 0 et 1) ou le nombre de lignes à dédier au test
        random_state (int or None): seed pour la reproductibilité

    Returns:
        train_df, test_df: la donnée à entraîner, la donnée à tester 
    """
    
    if not 0 <= test_size <= 1:
        raise ValueError("La taille du test doit être entre 0 et 1")

    if test_size > len(df):
        raise ValueError("La taille du test ne peut pas être supérieure à la taille de la dataframe")

    if random_state is not None:
        random.seed(random_state)

    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist() 
    test_indices = random.sample(population=indices, k=test_size)
    
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

def main():
    # 1. Charger, Lire, Visualiser, Préparer et Répartir les Données
    
    df = load_data('data/Iris.csv')

    df = check_data_validity(df)

    # visualize_initial_data(df)
    # seaborn_split(df)

    # Répartir les données en données d'entraînement et données de test
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)

    # Séparer les caractéristiques et les étiquettes
    X_train, Y_train = train_df.drop('species', axis=1), train_df['species']
    X_test, Y_test = test_df.drop('species', axis=1), test_df['species']
    
    # 2, 3, 4. Entraîner, Prédire et Évaluer le Modèle

    classifier = TreeClassifier()

    # Entraîner le modèle avec les données d'entraînement
    classifier.train(train_df, 'species')

    # Prédire les étiquettes pour les données de test
    predictions = classifier.predict(X_test)

    # Évaluer les performances du modèle
    accuracy = classifier.evaluate(predictions, Y_test)
    print(f"Précision: {accuracy * 100}%\n")

    # Afficher l'arbre de décision
    
    # Pour un affichage graphique avec graphviz et pydotplus
    # classifier.display_decision_tree()
    
    # Affichage textuel
    print("----- Arbre de décision -----\n")
    classifier.display_tree_textual()

if __name__ == "__main__":
    main()

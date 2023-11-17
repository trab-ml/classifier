from my_module.tree_classifier import TreeClassifier
import numpy as np
import pandas as pd
import random

def load_data(file_path):
    # Lit le fichier CSV

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

def check_data_validity(data):
    """
    Prend l'objet réprésentant la donnée en paramètre, 
    vérifie qu'il n'y a pas de valeurs nulles et supprime 
    les colonnes inutiles.

    Args:
        data (object): la donnée à traiter

    Returns:
        void
    """
    print("\ndf.describe() => \n")
    data.describe()

    print()

    print("\ndf.info() => \n")
    data.info()

    # supprime la colonne 'Id' qui ne sera pas utile
    data = data.drop('Id', axis=1)

    # Les 10 premières lignes 
    print("data.head(10) => \n")
    data.head(10)

    print()

def train_test_split(df, test_size):
    """
    Prend une dataframe en paramètre et le pourcentage (ou le nombre de lignes) de la dataframe à dédier au tester.

    Args:
        df (object): la donnée à séparer

    Returns:
        train_df, test_df: la donnée à entraîner, la donnée à tester 
    """

    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
        # print(test_size)

    # liste dans la variable indices les indices de la dataframe df
    indices = df.index.tolist() 
    test_indices = random.sample(population=indices, k=test_size)
    
    test_df = df.loc[test_indices] # Affecte à test_df les indices de la liste test_indices
    train_df = df.drop(test_indices) # Supprime les indices contenus dans test_df de train_df
    
    return train_df, test_df

def main():
    # 1. Charge les données
    df = load_data('data/Iris.csv')

    # 2. Vérifie les données
    check_data_validity(df)

    # 3. Visualise les données (si nécessaire)

    random.seed(0) # pour garder les mêmes valeurs le temps de cette instance d'exécution
    train_df, test_df = train_test_split(df, 0.3)

    # 4. Prépare les données
    X_train = df.drop('species', axis=1)  # Affecte à X toutes les colonnes sauf la cible
    y_train = df['species']  # Affecte à y la colonne cible

    # 5. Les données sont elles pures? classify : best split
    # if data is pure, classify
    # else if there are potential splits, perform splits
    # calculate entropy, information gain, and determine the best split

    # 7. Crée une instance de TreeClassifier
    tree_classifier = TreeClassifier()

    # 8. Entraîne le modèle
    tree_classifier.fit(X, y)

    # 9. Effectue des prédictions sur de nouvelles données
    # test_data = load_data('data/test_data.csv')
    # predictions = tree_classifier.predict(test_data)

    # 10. Affiche les prédictions
    # print(predictions)


if __name__ == "__main__":
    main()

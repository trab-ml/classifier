import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_initial_data(df):
    """
    Visualiser les données initiales avec un scatter plot.

    Args:
        df (pandas.DataFrame): La dataframe contenant les données.

    Returns:
        None
    """
    # Définir une couleur pour chaque espèce
    couleurs = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}

    # Utiliser la colonne 'species' pour colorer le scatter plot
    sns.scatterplot(x='petal_width', y='petal_length', hue='species', palette=couleurs, data=df)
    
    plt.xlabel("petal_width")
    plt.ylabel("petal_length")
    plt.title("Diagramme de dispersion de la largeur du pétale par rapport à la longueur du pétale")
    plt.legend()
    plt.show()

def seaborn_split(df):
    """
    Visualise les données divisées en utilisant un scatter plot.

    Args:
        df (pandas.DataFrame): La dataframe d'entrée contenant les données divisées.

    Returns:
        None
    """
    # Supposons que 'species', 'petal_width' et 'petal_length' sont les noms de colonnes dans votre dataframe
    plotting_df = pd.DataFrame(df, columns=['species', 'petal_width', 'petal_length'])

    # Créer un scatter plot en utilisant lmplot de seaborn
    sns.lmplot(data=plotting_df, x="petal_width", y="petal_length", hue="species", fit_reg=False)

    # Ajouter une ligne verticale pour représenter la valeur de division (0.8 dans ce cas)
    valeur_division = 0.8
    plt.vlines(x=valeur_division, ymin=1, ymax=7)

    # Définir la limite de l'axe x pour une meilleure visualisation
    plt.xlim(0, 2.6)

    # Afficher le plot
    plt.show()
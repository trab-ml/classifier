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
    plt.title("Dispersion de la largeur du pétale par rapport à la longueur du pétale")
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

    plotting_df = pd.DataFrame(df, columns=['species', 'petal_width', 'petal_length'])

    sns.lmplot(data=plotting_df, x="petal_width", y="petal_length", hue="species", fit_reg=False)

    valeur_division = 0.8
    
    plt.vlines(x=valeur_division, ymin=1, ymax=7)
    plt.xlim(0, 2.6)
    plt.show()
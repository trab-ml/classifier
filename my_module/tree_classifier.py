import numpy as np
import pandas as pd

def calculate_entropy(labels):
    """
    Prend un vecteur de labels en paramètre et calcule l'entropie de ce vecteur.
    
    Args:
        labels (array): vecteur de labels
    
    Returns:
        entropy (float): l'entropie du vecteur de labels
    """
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def calculate_information_gain(data, feature_name, target_name):
    """
    Prend la donnée, le nom de la feature et le nom de la target 
    en paramètre et calcule le gain d'information.
    
    Args:
        data (pandas.DataFrame): la donnée
        feature_name (str): le nom de la feature
        target_name (str): le nom de la target
    
    Returns:
        information_gain (float): le gain d'information
    """
    # Calcul de l'entropie initiale
    initial_entropy = calculate_entropy(data[target_name])
    print(f"Initial Entropy: {initial_entropy}")

    # Calcul de l'entropie après split
    values = data[feature_name].unique()
    new_entropy = 0
    for value in values:
        subset = data[data[feature_name] == value]
        weight = len(subset) / len(data[feature_name])
        entropy = calculate_entropy(subset[target_name])
        print(f"Subset Entropy ({feature_name} = {value}): {entropy}")
        new_entropy += weight * entropy

    # Calcul du gain d'information
    information_gain = initial_entropy - new_entropy
    
    return information_gain

def find_best_split(data, target_name):
    """
    Prend la donnée et le nom de la target en paramètre et trouve la meilleure feature
    et le meilleur gain d'information.
    
    Args:
        data (pandas.DataFrame): la donnée
        target_name (str): le nom de la target
        
    Returns:
        best_feature (str): le nom de la meilleure feature
        best_gain (float): le meilleur gain d'information
    """
    features = [col for col in data.columns if col != target_name]
    best_feature = None
    best_gain = -1

    for feature in features:
        gain = calculate_information_gain(data, feature, target_name)
        print(f"Feature: {feature}, Gain: {gain}\n")
        if gain > best_gain:
            best_gain = gain
            best_feature = feature

    print(f"Best Feature: {best_feature}, Best Gain: {best_gain}\n")

    return best_feature, best_gain

# Un classifieur d'arbre de décision.
class TreeClassifier:

    def __init__(self):
        self.tree = None

    def train(self, data, target_name):
        """
        Prend la donnée et le nom de la target en paramètre et construit un arbre de décision.
        Args:
            data (pandas.DataFrame): la donnée
            target_name (str): le nom de la target
        
        Returns:
            None
        """
        # Lorsque toutes les instances ont la même classe, on retourne un noeud feuille
        if len(np.unique(data[target_name])) == 1:
            self.tree = {'class': data[target_name].iloc[0]}
            return

        best_feature, _ = find_best_split(data, target_name)

        # Crée un noeud de décision 
        self.tree = {'feature': best_feature, 'branches': {}}

        # Construction récursive de l'arbre
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            if not subset.empty:  # Le sous-ensemble est-il vide?
                subtree_classifier = TreeClassifier()
                subtree_classifier.train(subset, target_name)
                self.tree['branches'][value] = subtree_classifier.tree
            else:
                print("Le sous-ensemble est vide")
        
    def predict(self, data):
        """
        Prend la donnée en paramètre et retourne les prédictions.
        
        Args:
            data (pandas.DataFrame): la donnée
        
        Returns:
            predictions (list): les prédictions
        """
        predictions = []
        for _, instance in data.iterrows():
            tree = self.tree
            while 'class' not in tree:
                feature_value = instance[tree['feature']]
                if feature_value not in tree['branches']:
                    predictions.append(None)
                    break
                tree = tree['branches'][feature_value]
            else:
                predictions.append(tree['class'])
                
        return predictions

    def evaluate(self, predicted_labels, true_labels):
        """
        Prend les prédictions et les vraies valeurs en paramètre et retourne l'accuracy.
        
        Args:
            predicted_labels (list): les prédictions
            true_labels (list): les vraies valeurs
        
        Returns:
            accuracy (float): la précision
        """
        correct_predictions = sum(pred == true for pred, true in zip(predicted_labels, true_labels))
        accuracy = correct_predictions / len(true_labels)
        return accuracy

    def display_tree(self, tree=None, indent=0):
        """ 
        Prend un arbre de décision en paramètre et affiche l'arbre.
        
        Args:
            tree (dict): l'arbre de décision
            indent (int): l'indentation
        
        Returns:
            None
        """
        if tree is None:
            tree = self.tree
        if 'class' in tree:
            print(f"{'  ' * indent}Class: {tree['class']}")
        else:
            print(f"{'  ' * indent}Feature: {tree['feature']}")
            for value, subtree in tree['branches'].items():
                print(f"{'  ' * (indent + 1)}Value {tree['feature']} = {value}:")
                self.display_tree(subtree, indent + 2)

import numpy as np
import pandas as pd
from graphviz import Digraph

def calculate_entropy(labels):
    """
    Takes a vector of labels as parameter and calculates the entropy.
    
    Args:
        labels (array): Vector of labels
    
    Returns:
        entropy (float): The entropy of the labels
    """
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def calculate_information_gain(data, feature_name, target_name):
    """
    Takes the data, the name of the feature and the name of the target as parameters
    and calculates the information gain.
    
    Args:
        data (pandas.DataFrame): the data
        feature_name (str): the name of the feature
        target_name (str): the name of the target
    
    Returns:
        information_gain (float): the information gain
    """
    # Calculation of the initial entropy
    initial_entropy = calculate_entropy(data[target_name])
    print(f"Initial Entropy: {initial_entropy}")

    # Calculation of the new entropy
    values = data[feature_name].unique()
    new_entropy = 0
    for value in values:
        subset = data[data[feature_name] == value]
        weight = len(subset) / len(data[feature_name])
        entropy = calculate_entropy(subset[target_name])
        print(f"Subset Entropy ({feature_name} = {value}): {entropy}")
        new_entropy += weight * entropy

    # Calculation of the information gain
    information_gain = initial_entropy - new_entropy
    
    return information_gain

def find_best_split(data, target_name):
    """
    Takes the data and the name of the target as parameters and returns the best feature
    and the best information gain.
    
    Args:
        data (pandas.DataFrame): the data
        target_name (str): the name of the target
        
    Returns:
        best_feature (str): the best feature
        best_gain (float): the best information gain
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

# A class to represent a decision tree classifier
class TreeClassifier:

    def __init__(self):
        self.tree = None

    def train(self, data, target_name):
        """
        Takes the data and the name of the target as parameters and trains the decision tree.
            
        Args:
            data (pandas.DataFrame): the data
            target_name (str): the name of the target
        
        Returns:
            None
        """

        if len(np.unique(data[target_name])) == 1:
            self.tree = {'class': data[target_name].iloc[0]}
            return

        best_feature, _ = find_best_split(data, target_name)

        # Create a node for the best feature
        self.tree = {'feature': best_feature, 'branches': {}}

        # Recursive construction of the tree
        for value in data[best_feature].unique():
            subset = data[data[best_feature] == value]
            if not subset.empty:  # Is the subset empty?
                subtree_classifier = TreeClassifier()
                subtree_classifier.train(subset, target_name)
                self.tree['branches'][value] = subtree_classifier.tree
            else:
                print("The subset is empty!")
        
    def predict(self, data):
        """
        Takes the data as parameter and returns the predictions.
        
        Args:
            data (pandas.DataFrame): The data to predict
        
        Returns:
            predictions (list): predictions
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
        Takes the predicted labels and the true labels as parameters and returns the accuracy.
        
        Args:
            predicted_labels (list): predicted labels
            true_labels (list): true labels
        
        Returns:
            accuracy (float): prediction accuracy
        """
        correct_predictions = sum(pred == true for pred, true in zip(predicted_labels, true_labels))
        accuracy = correct_predictions / len(true_labels)
        return accuracy

    def display_tree_graphviz(self, tree=None, dot=None):
        """ 
        Takes a decision tree as parameter and returns a Graphviz object to display the tree.
        
        Args:
            tree (dict): The decision tree
            dot (graphviz.Digraph): Graphviz object to display the tree
        
        Returns:
            dot (graphviz.Digraph): An Graphviz object to display the tree
        """
        if tree is None:
            tree = self.tree
            dot = Digraph(comment='Decision Tree')

        if 'class' in tree:
            # If it's a leaf node (class), simply display the class
            dot.node(str(tree['class']), shape='ellipse')
        else:
            # If it's a decision node, display the feature and the branches
            dot.node(tree['feature'], shape='box')
            for value, subtree in tree['branches'].items():
                dot = self.display_tree_graphviz(subtree, dot)
                # Add an edge between the decision node and the subtree
                if 'feature' in subtree:
                    dot.edge(tree['feature'], subtree['feature'], label=str(value))

        return dot

    def display_decision_tree(self):
        """
        Displays the decision tree.
        """
        dot = self.display_tree_graphviz()
        dot.render('decision_tree', format='png', cleanup=True, view=True)
        
    def display_tree_textual(self, tree=None, indent=0):
        """ 
        Takes a decision tree as parameter and returns a textual representation of the tree.
        
        Args:
            tree (dict): decision tree
            indent (int): indentation level
        
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
                self.display_tree_textual(subtree, indent + 2)

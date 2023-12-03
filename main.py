from my_module.tree_classifier import TreeClassifier
from my_module.data_visualizer import seaborn_split, visualize_initial_data
import numpy as np
import pandas as pd
import random
import time
import psutil

def load_data(file_path):
    """
    Load properly the CSV file.
    
    Args:
        file_path: path to the CSV file

    Returns:
        the loaded data
    """

    try:
        df = pd.read_csv(file_path)

    except FileNotFoundError:
        print("File not found !.")

    except pd.errors.EmptyDataError:
        print("Empty file !.")

    except pd.errors.ParserError as e:
        print(f"An Error occured while parsing the file: {e}")

    except Exception as e:
        print(f"Unexpected error: {e}")

    return df

def check_data_validity(df):
    """
    Take a dataframe as parameter, check that there are no null values 
    and remove unnecessary columns.

    Args:
        df (dataframe): the data to check

    Returns:
        df (dataframe): a valid dataframe
    """
    # print("\ndf.describe() => \n")
    # print(df.describe())

    # print("\ndf.info() => \n")
    # df.info()

    # remove unnecessary columns
    df = df.drop('Id', axis=1)

    # print("\ndf.head(10) => \n")
    # print(df.head(10))
    return df

def train_test_split(df, test_size, random_state=None):
    """
    Take a dataframe as parameter, the percentage (or the number of rows)
    of the dataframe to dedicate to the test and respectively return the data 
    to train (train_data) and the data to test (test_data).

    Args:
        df (pandas.DataFrame): the data to split
        test_size (float): the percentage (or the number of rows) of 
        the dataframe to dedicate to the test

    Returns:
        train_df, test_df: the data to train and the data to test
    """
    
    if not 0 <= test_size <= 1:
        raise ValueError("The test size must be between 0 and 1")

    if test_size > len(df):
        raise ValueError("The test size must not be greater than the dataframe length")

    if random_state is not None:
        random.seed(random_state)

    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist() 
    test_indices = random.sample(population=indices, k=test_size)
    
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

import time
import psutil

def measure_time_and_memory(func):
    def wrapper(*args, **kwargs):
        # Measure the execution time
        start_time = time.time()

        # Measure the memory usage before the function call
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # in Mo

        # Execute the function to measure
        result = func(*args, **kwargs)

        # Measure the memory usage after the function call
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # in Mo

        # Measure the execution time
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"\nTemps d'exécution : {execution_time} secondes")
        print(f"Utilisation de la mémoire : {end_memory - start_memory} Mo")

        return result

    return wrapper

@measure_time_and_memory
def main():
    # 1. Load, Read, Visualize, Prepare and Split the Data
    
    df = load_data('data/Iris.csv')

    df = check_data_validity(df)

    # visualize_initial_data(df)
    # seaborn_split(df)

    # Distribute the data into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=0)

    # Seperate the features and the labels
    X_train, Y_train = train_df.drop('species', axis=1), train_df['species']
    X_test, Y_test = test_df.drop('species', axis=1), test_df['species']
    
    # 2, 3, 4. Train, Predict and Evaluate the Model

    classifier = TreeClassifier()

    # Train the model with the train data
    classifier.train(train_df, 'species')

    # Predict the labels of the test data
    predictions = classifier.predict(X_test)

    # Evaluate the model
    accuracy = classifier.evaluate(predictions, Y_test)
    print(f"Précision: {accuracy * 100}%\n")

    # Print of Decsion Tree
    
    # Graphic display with graphviz and pydotplus
    # classifier.display_decision_tree()
    
    # Textual display
    # print("----- Arbre de décision -----\n")
    # classifier.display_tree_textual()

if __name__ == "__main__":
    main()

# Apprentissage automatique - Classifieur

| Étape | Tâche | Définition | Fonctionnalités/Logiques à Implémenter |
|-------|-------|------------|----------------------------------------|
| 0     | Prétraitement des données | Cette étape comprend la lecture des données, le nettoyage et la vérification de la conformité. | - |
| 0.1   | Lecture des données | Charger les données à partir d'une source externe (par exemple, un fichier CSV) dans une structure de données utilisable. | Implémentation de la fonction `lire_charger_donnees` pour lire et charger les données à partir d'un fichier CSV. |
| 0.2   | Nettoyage des données | Identifier et traiter les valeurs manquantes, les valeurs aberrantes ou d'autres anomalies dans les données. | - |
| 0.3   | Vérification de la conformité | S'assurer que les données sont dans un format approprié, et qu'elles ne contiennent pas d'incohérences majeures. | - |
| 1     | Initialisation du classifieur | Création de la classe `TreeDecisionClassifier`. Initialisation des attributs (`max_depth`, `tree`). | - |
| 2     | Entraînement du modèle | Implémentation de la méthode `fit` pour construire l'arbre de décision à partir des données d'entraînement. Utilisation de la méthode `_build_tree` pour récursivement construire l'arbre. Choix des critères de division basés sur l'entropie. | - |
| 3     | Calcul de l'entropie | L'entropie mesure le niveau de désordre ou d'incertitude dans un ensemble de données. Plus l'entropie est élevée, plus l'ensemble de données est mélangé. Le but est de minimiser l'entropie en trouvant des divisions qui réduisent l'incertitude sur les classes des données. | Implémentation de la méthode `_calculate_entropy` pour calculer l'entropie d'un groupe de données. |
| 4     | Division des données | La division des données consiste à séparer un ensemble de données en deux groupes en fonction d'une valeur seuil sur une caractéristique. Cette division est effectuée pour réduire l'entropie des groupes résultants. | Implémentation de la méthode `_split` pour diviser les données en deux groupes en fonction d'une valeur seuil. |
| 5     | Recherche de la meilleure division | La meilleure division est celle qui maximise la réduction d'entropie, c'est-à-dire celle qui minimise l'incertitude dans les groupes résultants. | Implémentation de la méthode `_find_best_split` pour trouver la meilleure division basée sur l'entropie. |
| 6     | Construction de l'arbre | Construction récursive de l'arbre en choisissant la meilleure division à chaque étape pour minimiser l'entropie. | Implémentation de la méthode `_build_tree` pour construire récursivement l'arbre. |
| 7     | Prédiction d'une instance | Faire une prédiction pour une instance donnée en traversant l'arbre entraîné. | Implémentation de la méthode `_predict_instance` pour faire une prédiction pour une instance donnée en traversant l'arbre. |
| 8     | Prédiction sur l'ensemble de test | Faire des prédictions sur l'ensemble de test en utilisant l'arbre construit. | Implémentation de la méthode `predict` pour faire des prédictions sur l'ensemble de test. |
| 9     | Évaluation du modèle | Comparaison des prédictions avec les véritables étiquettes pour évaluer la précision du modèle. Calcul de l'exactitude. | - |

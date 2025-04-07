# Prédiction de résultats de matchs de football

Ce projet utilise des techniques d'apprentissage automatique pour prédire les résultats des matchs de football en se basant sur des statistiques d'équipes et de joueurs.

## 📋 Description du projet

Le système analyse les données historiques des performances des équipes à domicile et à l'extérieur, ainsi que les statistiques individuelles des joueurs, pour prédire si un match se terminera par une victoire à domicile, un match nul ou une victoire à l'extérieur.

Deux approches de modélisation sont implémentées :
1. **Modèle de base** (`foot-prediction.py`) : utilise la régression logistique avec un ensemble limité de caractéristiques
2. **Modèle avancé** (`enhanced_prediction.py`) : utilise XGBoost avec une validation croisée et une optimisation d'hyperparamètres

## 🗄️ Structure des données

Le projet exploite quatre ensembles de données principaux :

- **Statistiques d'équipe à domicile** : performances collectives des équipes jouant à domicile
- **Statistiques d'équipe à l'extérieur** : performances collectives des équipes jouant à l'extérieur
- **Statistiques des joueurs à domicile** : performances individuelles des joueurs des équipes à domicile
- **Statistiques des joueurs à l'extérieur** : performances individuelles des joueurs des équipes à l'extérieur

Ces données sont divisées en ensembles d'entraînement et de test.

## 🔍 Caractéristiques principales

### Modèle de base (`foot-prediction.py`)
- Utilisation d'un nombre limité de caractéristiques (passes précises, tirs hors cible, passes décisives)
- Régression logistique avec validation croisée
- Prétraitement de base des données
- Visualisation des résultats avec matrices de confusion et d'importance des caractéristiques

### Modèle avancé (`enhanced_prediction.py`)
- Prétraitement robuste des données
  - Gestion des valeurs manquantes avec la médiane
  - Traitement des valeurs aberrantes avec écrêtage
  - Mise à l'échelle avec RobustScaler
- Utilisation du classifieur XGBoost pour la prédiction multiclasse
- Validation croisée stratifiée (5 plis)
- Test de plusieurs combinaisons d'hyperparamètres
- Optimisation pour l'ensemble de données déséquilibrées

## 🚀 Installation et utilisation

### Prérequis
- Python 3.x
- Bibliothèques requises : pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn

### Installation
```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/Prediction-de-resultat-de-match.git
cd Prediction-de-resultat-de-match

# Installer les dépendances
pip install -r requirements.txt
```

### Exécution
Pour le modèle de base :
```bash
python foot-prediction.py
```

Pour le modèle avancé :
```bash
python enhanced_prediction.py
```

## 📊 Évaluation du modèle

Le modèle avancé utilise l'accuracy comme métrique principale d'évaluation, calculée via une validation croisée stratifiée à 5 plis. Cette approche permet de s'assurer que le modèle fonctionne bien sur différentes partitions des données et réduit les risques de surajustement.

## 📝 Détails techniques

### Prétraitement des données
- **Nettoyage** : remplacement des valeurs manquantes, infinies et aberrantes
- **Sélection des caractéristiques** : identification des colonnes communes entre ensembles d'entraînement et de test
- **Mise à l'échelle** : application de RobustScaler pour réduire l'influence des valeurs extrêmes

### Entraînement du modèle
- **Validation croisée** : StratifiedKFold pour préserver la distribution des classes
- **Hyperparamètres** : test de plusieurs configurations pour optimiser les performances
- **Évaluation** : mesure d'accuracy pour sélectionner le meilleur modèle

### Prédiction
- Transformation des données de test avec le même prétraitement que les données d'entraînement
- Prédiction des probabilités pour chaque classe (victoire à domicile, match nul, victoire à l'extérieur)
- Création d'un fichier de soumission au format attendu

## 🛠️ Améliorations possibles

- Ingénierie de caractéristiques plus poussée
- Exploration d'autres algorithmes (réseaux de neurones, ensembles plus complexes)
- Optimisation plus fine des hyperparamètres (GridSearchCV ou RandomizedSearchCV)
- Analyse plus approfondie des caractéristiques importantes
- Intégration de données externes (météo, historique des rencontres, etc.)

## 📈 Performances

Le modèle avancé obtient de meilleures performances que le modèle de base grâce à :
- Un prétraitement plus robuste des données
- L'utilisation d'un algorithme plus puissant (XGBoost)
- L'optimisation des hyperparamètres
- La gestion plus efficace des valeurs manquantes et aberrantes

## 📄 Licence

[Insérer information de licence]

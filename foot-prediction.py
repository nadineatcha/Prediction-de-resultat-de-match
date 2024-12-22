# Importation des bibliothèques nécessaires
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Vérifie le répertoire de travail et change si nécessaire
print("Répertoire de travail actuel :", os.getcwd())
os.chdir('/Users/nadine/Desktop/Ecole/Machinelearning/Dossier challenge')

# Chargement des fichiers CSV
try:
    away_team_stats = pd.read_csv("TrainX/train_away_team_statistics_df.csv")
    home_team_stats = pd.read_csv("TrainX/train_home_team_statistics_df.csv")
    results = pd.read_csv("Y.csv")
    home_player_stats = pd.read_csv("TrainX/train_home_player_statistics_df.csv")
    away_player_stats = pd.read_csv("TrainX/train_away_player_statistics_df.csv")
    print("Fichiers chargés avec succès.")
except FileNotFoundError as e:
    print(f"Erreur de chargement des fichiers : {e}")
    raise

# Exploration des données
print("Colonnes 'home_team_stats' :", home_team_stats.columns)
print("Colonnes 'away_team_stats' :", away_team_stats.columns)
print("Statistiques des joueurs à domicile :", home_player_stats.head())
print("Statistiques des joueurs à l'extérieur :", away_player_stats.head())

# Vérification des valeurs manquantes
print("Valeurs manquantes dans 'home_team_stats':\n", home_team_stats.isnull().sum())
print("Valeurs manquantes dans 'away_team_stats':\n", away_team_stats.isnull().sum())

# Fusion des données avec les résultats cibles
home_team_stats = home_team_stats.merge(results, left_index=True, right_index=True)

# Préparation des données pour la modélisation
X_home = home_player_stats[['PLAYER_SHOTS_OFF_TARGET_5_last_match_std', 
                            'PLAYER_ACCURATE_PASSES_season_sum', 
                            'PLAYER_ASSISTS_season_sum']]
X_away = away_player_stats[['PLAYER_SHOTS_OFF_TARGET_5_last_match_std', 
                            'PLAYER_ACCURATE_PASSES_season_sum', 
                            'PLAYER_ASSISTS_season_sum']]

# Gestion des valeurs manquantes
X_home.fillna(0, inplace=True)
X_away.fillna(0, inplace=True)

# Création de la variable cible (MATCH_RESULT)
if 'MATCH_RESULT' not in home_player_stats.columns:
    home_player_stats['MATCH_RESULT'] = (home_player_stats['PLAYER_ACCURATE_PASSES_season_sum'] > 10).astype(int)

if 'MATCH_RESULT' not in away_player_stats.columns:
    away_player_stats['MATCH_RESULT'] = (away_player_stats['PLAYER_ACCURATE_PASSES_season_sum'] > 10).astype(int)

y_home = home_player_stats['MATCH_RESULT']
y_away = away_player_stats['MATCH_RESULT']

# Fusion des données d'entraînement
X = pd.concat([X_home, X_away], axis=0)
y = pd.concat([y_home, y_away], axis=0)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Suppression des colonnes fortement corrélées
columns_to_drop = ['PLAYER_ACCURATE_PASSES_season_sum', 'PLAYER_SHOTS_OFF_TARGET_5_last_match_std']
X_train = X_train.drop(columns=columns_to_drop, errors='ignore')
X_test = X_test.drop(columns=columns_to_drop, errors='ignore')

# Mise à l'échelle des données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entraînement d'un modèle de régression logistique
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train_scaled, y_train)

# Évaluation du modèle
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Précision du modèle : {accuracy}")
print("Matrice de confusion :\n", conf_matrix)
print(classification_report(y_test, y_pred))

# Validation croisée
cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print("Scores de validation croisée :", cross_val_scores)
print("Précision moyenne :", cross_val_scores.mean())

# Importance des caractéristiques
if hasattr(model, 'coef_'):
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.coef_[0]
    }).sort_values(by='Importance', ascending=False)
elif hasattr(model, 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
else:
    feature_importances = None

if feature_importances is not None:
    print("Importance des caractéristiques :\n", feature_importances)

    # Visualisation
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importances, x='Importance', y='Feature', palette='viridis')
    plt.title('Importance des Caractéristiques')
    plt.tight_layout()
    plt.show()

# Visualisation des performances
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False)
plt.title("Matrice de Confusion")
plt.xlabel("Prédictions")
plt.ylabel("Vérités")
plt.show()

# Matrice de corrélation
correlation_matrix = pd.concat([X_train, y_train], axis=1).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()

# Chargement de l'ensemble de test
test_data = pd.read_csv("/Users/nadine/desktop/Ecole/Machinelearning/Dossier challenge/TrainX/Test_Data/test_away_player_statistics_df.csv") 

# Vérifier les premières lignes pour comprendre la structure
print("Aperçu des données de test :")
print(test_data.head())

# Préparer les données de test
# Vérifiez si toutes les colonnes utilisées pour l'entraînement sont dans l'ensemble de test
columns_used_in_training = X_train.columns  # Colonnes utilisées pour l'entraînement
missing_columns = [col for col in columns_used_in_training if col not in test_data.columns]

if missing_columns:
    print(f"Les colonnes suivantes sont manquantes dans l'ensemble de test : {missing_columns}")
    for col in missing_columns:
        test_data[col] = 0  # Ajout de colonnes manquantes avec des valeurs par défaut

# Nettoyer les données de test
test_data_cleaned = test_data[columns_used_in_training].copy()  # Conserver uniquement les colonnes nécessaires
test_data_cleaned = test_data_cleaned.fillna(0)  # Remplir les valeurs manquantes

# Appliquer la mise à l'échelle (StandardScaler)
scaler = StandardScaler()
test_data_scaled = scaler.fit_transform(test_data_cleaned)

# Prédictions sur les données de test
test_predictions = model.predict(test_data_scaled)

# Créer un fichier de soumission
submission = pd.DataFrame({
    'ID': test_data['ID'],  # Assurez-vous que la colonne 'ID' est présente dans test_data
    'prediction': test_predictions
})


# Prendre la prédiction majoritaire pour chaque ID
submission = submission.groupby('ID', as_index=False)['prediction'].agg(lambda x: x.mode()[0])

# Vérifier l'aperçu
print("Fichier de soumission après regroupement :")
print(submission.head())

submission.rename(columns={'id': 'ID', 'Prediction': 'prediction'}, inplace=True)

# Sauvegarder dans un fichier CSV sans index
submission.to_csv("submission.csv", index=False)

submission = submission.dropna(subset=['prediction'])


# Moyenne des prédictions par ID
submission = submission.groupby('ID', as_index=False)['prediction'].mean()

# Convertir la moyenne en valeurs binaires (0 ou 1)
submission['prediction'] = (submission['prediction'] > 0.5).astype(int)

# Sauvegarder le fichier corrigé
submission.to_csv("submission.csv", index=False)







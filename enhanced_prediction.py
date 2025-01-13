import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def get_common_columns(train_df, test_df):
    """Retourne les colonnes communes entre train et test"""
    train_cols = set(train_df.select_dtypes(include=[np.number]).columns)
    test_cols = set(test_df.select_dtypes(include=[np.number]).columns)
    return sorted(list(train_cols.intersection(test_cols)))

def clean_and_prepare_features(df):
    """Nettoie et prépare les features"""
    df_clean = df.copy()
    
    # Remplacement des valeurs infinies et NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Pour chaque colonne numérique
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Remplacement des NaN par la médiane
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Gestion des valeurs extrêmes
        q1, q3 = df_clean[col].quantile([0.01, 0.99])
        df_clean[col] = df_clean[col].clip(lower=q1, upper=q3)
    
    return df_clean

def train_with_cv(X, y, n_splits=5):
    """Entraîne plusieurs modèles avec différents hyperparamètres"""
    param_combinations = [
        {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
        },
        {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
        },
        {
            'n_estimators': 300,
            'max_depth': 8,
            'learning_rate': 0.01,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
        }
    ]
    
    best_score = 0
    best_model = None
    best_predictions = None
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for params in param_combinations:
        print(f"\nTest des paramètres: {params}")
        scores = []
        predictions = np.zeros((len(X), 3))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"Fold {fold}/{n_splits}")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                random_state=42,
                **params
            )
            
            model.fit(X_train, y_train)
            val_pred = model.predict(X_val)
            score = accuracy_score(y_val, val_pred)
            scores.append(score)
            
            predictions[val_idx] = model.predict_proba(X_val)
        
        avg_score = np.mean(scores)
        print(f"Score moyen: {avg_score:.4f} (±{np.std(scores):.4f})")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = model
            best_predictions = predictions
            print(f"Nouveau meilleur score!")
    
    return best_model, best_predictions, best_score

def main():
    try:
        print("Chargement des données...")
        # Données d'entraînement et test
        train_home = pd.read_csv("Train_Data/train_home_team_statistics_df.csv")
        train_away = pd.read_csv("Train_Data/train_away_team_statistics_df.csv")
        test_home = pd.read_csv("Test_Data/test_home_team_statistics_df.csv")
        test_away = pd.read_csv("Test_Data/test_away_team_statistics_df.csv")
        y_data = pd.read_csv("Y.csv")

        # Trouver les colonnes communes
        home_cols = get_common_columns(train_home, test_home)
        away_cols = get_common_columns(train_away, test_away)
        
        print(f"\nNombre de features communes: Home={len(home_cols)}, Away={len(away_cols)}")
        
        # Nettoyage des features
        print("\nNettoyage des features...")
        home_train_clean = clean_and_prepare_features(train_home[home_cols])
        away_train_clean = clean_and_prepare_features(train_away[away_cols])
        home_test_clean = clean_and_prepare_features(test_home[home_cols])
        away_test_clean = clean_and_prepare_features(test_away[away_cols])
        
        # Combinaison des features
        X = pd.concat([home_train_clean, away_train_clean], axis=1)
        X_test = pd.concat([home_test_clean, away_test_clean], axis=1)
        
        # Vérification des colonnes
        print(f"\nColonnes d'entraînement: {X.shape[1]}")
        print(f"Colonnes de test: {X_test.shape[1]}")
        
        # Préparation de la target
        y = np.argmax(y_data[['HOME_WINS', 'DRAW', 'AWAY_WINS']].values, axis=1)
        
        print(f"\nDimensions des données: {X.shape}")
        print(f"Distribution des classes:\n{pd.Series(y).value_counts(normalize=True)}")
        
        # Scaling des features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("\nEntraînement des modèles avec validation croisée...")
        best_model, cv_predictions, best_score = train_with_cv(X_scaled, y)
        print(f"\nMeilleur score CV: {best_score:.4f}")
        
        # Prédictions sur l'ensemble de test
        print("\nGénération des prédictions finales...")
        X_test_scaled = scaler.transform(X_test)
        predictions = best_model.predict(X_test_scaled)
        
        # Création du fichier de soumission
        submission = pd.DataFrame({
            'ID': test_home['ID'],
            'HOME_WINS': (predictions == 0).astype(int),
            'DRAW': (predictions == 1).astype(int),
            'AWAY_WINS': (predictions == 2).astype(int)
        })
        
        submission.to_csv('submission_enhanced.csv', index=False)
        print("\nPrédictions sauvegardées dans 'submission_enhanced.csv'")
        
    except Exception as e:
        print(f"\nErreur: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
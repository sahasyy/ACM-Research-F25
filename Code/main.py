# main logic for logistic regression binary classifier

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')


class GameChurnPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.feature_names = []

    def load_game1_data(self, filepath):
        """Load Game 1 data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            print(f"Game 1 data loaded: {df.shape}")
            return df
        except FileNotFoundError:
            print("Game 1 file not found")

    def load_game2_data(self, play_filepath, purchase_filepath=None):
        """Load Game 2 data from CSV files"""
        try:
            play_df = pd.read_csv(play_filepath)
            print(f"Game 2 play data loaded: {play_df.shape}")

            if purchase_filepath:
                purchase_df = pd.read_csv(purchase_filepath)
                print(f"Game 2 purchase data loaded: {purchase_df.shape}")
                # Merge on device.id (change)?
                df = pd.merge(play_df, purchase_df, on='device.id', how='left')
                # Fill missing purchase data with 0
                df['highestPrice'] = df['highestPrice'].fillna(0)
                df['purchaseCount'] = df['purchaseCount'].fillna(0)
            else:
                df = play_df
                df['highestPrice'] = 0
                df['purchaseCount'] = 0

            print(df)
            return df
        except FileNotFoundError:
            print("Game 2 files not found")

    def define_churn(self, df, churn_days=10):
        """Define churn based on lastPlayTime and observationPeriodEnd"""
        df_processed = df.copy()

        # Convert datetime columns
        datetime_cols = ['firstPlayTime', 'lastPlayTime']
        obs_col = 'individualOPEnd'

        for col in datetime_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_datetime(df_processed[col])

        # Convert observation period end to datetime
        df_processed[obs_col] = pd.to_datetime(df_processed[obs_col])

        # Print churn distribution
        churn_counts = df_processed['is_churned'].value_counts()
        churn_rate = churn_counts[1] / len(df_processed) * 100
        print(f"\nChurn Distribution:")
        print(f"Not Churned: {churn_counts[0]} ({100 - churn_rate:.1f}%)")
        print(f"Churned: {churn_counts[1]} ({churn_rate:.1f}%)")

        return df_processed

    def balance_dataset(self, X, y, method='smote', target_ratio=0.3):
        """
        Balance the dataset using various techniques

        Parameters:
        - method: 'smote', 'undersample', 'smoteenn', 'threshold_adjust'
        - target_ratio: desired ratio of minority class (0.3 = 30% churned)
        """
        print(f"\nOriginal distribution: {np.bincount(y)}")

        if method == 'smote':
            # SMOTE - Synthetic Minority Oversampling
            smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)

        elif method == 'undersample':
            # Random undersampling of majority class
            undersampler = RandomUnderSampler(sampling_strategy=target_ratio / (1 - target_ratio), random_state=42)
            X_balanced, y_balanced = undersampler.fit_resample(X, y)

        elif method == 'smoteenn':
            # SMOTE + Edited Nearest Neighbours
            smoteenn = SMOTEENN(sampling_strategy=target_ratio, random_state=42)
            X_balanced, y_balanced = smoteenn.fit_resample(X, y)

        elif method == 'threshold_adjust':
            # Keep original data but adjust decision threshold later
            return X, y

        else:
            raise ValueError("Method must be 'smote', 'undersample', 'smoteenn', or 'threshold_adjust'")

        print(f"Balanced distribution: {np.bincount(y_balanced)}")
        balance_rate = np.bincount(y_balanced)[1] / len(y_balanced) * 100
        print(f"New churn rate: {balance_rate:.1f}%")

        return X_balanced, y_balanced

    def find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal classification threshold using F1 score"""
        from sklearn.metrics import f1_score, precision_recall_curve

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        f1_scores = np.nan_to_num(f1_scores)  # Handle division by zero

        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

        return optimal_threshold, f1_scores[optimal_idx]

    def prepare_features(self, df, game_type='game1'):
        """Prepare features for modeling using only original features"""
        # Game 1 features
        if game_type == 'game1':
            feature_cols = [
                'activeDuration', 'bestScore', 'bestScoreIndex',
                'bestSubMeanCount', 'bestSubMeanRatio', 'consecutivePlayRatio',
                'meanScore', 'playCount', 'sdScore', 'worstScore'
            ]

        # Game 2 features
        elif game_type == 'game2':
            feature_cols = [
                'activeDuration', 'bestScore', 'bestScoreIndex',
                'bestSubMeanCount', 'bestSubMeanRatio', 'consecutivePlayRatio',
                'meanScore', 'playCount', 'sdScore', 'worstScore',
                'highestPrice', 'purchaseCount'
            ]

        # Select only existing columns
        available_cols = [col for col in feature_cols if col in df.columns]

        X = df[available_cols].copy()
        y = df['is_churned'].copy()

        # Handle missing values
        X = X.fillna(X.median())

        self.feature_names = available_cols
        return X, y

    def train_model(self, X, y, test_size=0.2, balance_method='smote', use_class_weight=False):
        """Train the logistic regression model with balancing options"""

        # Split the data first
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Balance training data if requested
        if balance_method != 'none':
            X_train_balanced, y_train_balanced = self.balance_dataset(X_train, y_train, balance_method)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_balanced)
        X_test_scaled = self.scaler.transform(X_test)

        # Configure model with class weights if requested
        if use_class_weight:
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            )
        else:
            self.model = LogisticRegression(random_state=42, max_iter=1000)

        # Train model
        self.model.fit(X_train_scaled, y_train_balanced)

        # Make predictions
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Find optimal threshold
        optimal_threshold, best_f1 = self.find_optimal_threshold(y_test, y_pred_proba)
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)

        print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.3f})")

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train_balanced,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'optimal_threshold': optimal_threshold
        }

    def evaluate_model(self, results):
        """Evaluate model performance"""
        y_test = results['y_test']
        y_pred = results['y_pred']
        y_pred_proba = results['y_pred_proba']

        print("=== Model Evaluation ===")
        print(f"AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        print(f"Optimal Threshold: {results['optimal_threshold']:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': abs(self.model.coef_[0])
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

        return feature_importance

    def plot_results(self, results, feature_importance):
        """Plot model results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # ROC Curve
        fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
        auc_score = roc_auc_score(results['y_test'], results['y_pred_proba'])

        axes[0, 0].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.4f})')
        axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 0].axvline(x=results['optimal_threshold'], color='red', linestyle='--', alpha=0.7,
                           label='Optimal Threshold')
        axes[0, 0].set_xlabel('False Positive Rate')
        axes[0, 0].set_ylabel('True Positive Rate')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Confusion Matrix
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
        axes[0, 1].set_title('Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')

        # Feature Importance
        top_features = feature_importance.head(12)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_xlabel('Absolute Coefficient Value')
        axes[1, 0].set_title('Top 10 Feature Importance')

        # Prediction Distribution
        axes[1, 1].hist(results['y_pred_proba'][results['y_test'] == 0],
                        alpha=0.7, bins=30, label='Non-Churned', density=True)
        axes[1, 1].hist(results['y_pred_proba'][results['y_test'] == 1],
                        alpha=0.7, bins=30, label='Churned', density=True)
        axes[1, 1].axvline(x=results['optimal_threshold'], color='red', linestyle='--', alpha=0.7,
                           label='Optimal Threshold')
        axes[1, 1].set_xlabel('Churn Probability')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Prediction Probability Distribution')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def predict_churn(self, df, game_type='game1'):
        """Predict churn for new data"""
        df_processed = self.define_churn(df)
        X, _ = self.prepare_features(df_processed, game_type)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict_proba(X_scaled)[:, 1]
        df_processed['churn_probability'] = predictions
        df_processed['predicted_churn'] = (predictions > 0.5).astype(int)

        return df_processed[['churn_probability', 'predicted_churn']]

    def compare_balancing_methods(self, X, y):
        """Compare different balancing methods"""
        methods = ['none', 'smote', 'undersample', 'smoteenn']
        results = {}

        print("=== Comparing Balancing Methods ===")

        for method in methods:
            print(f"\n--- {method.upper()} ---")

            # Train model with this balancing method
            model_results = self.train_model(X, y, balance_method=method)

            # Evaluate
            y_test = model_results['y_test']
            y_pred = model_results['y_pred']
            y_pred_proba = model_results['y_pred_proba']

            auc = roc_auc_score(y_test, y_pred_proba)
            from sklearn.metrics import f1_score, precision_score, recall_score

            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            results[method] = {
                'AUC': auc,
                'F1': f1,
                'Precision': precision,
                'Recall': recall
            }

            print(f"AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        print(f"\n=== Summary ===")
        print(comparison_df.round(4))

        return comparison_df

    def get_separate_rankings_original_features(self, game1_data, game2_data, balance_method='smote'):
        """Get separate rankings using original feature sets with balancing"""

        # Game 1 with 10 features
        print("\n=== GAME 1 ANALYSIS ===")
        game1_processed = self.define_churn(game1_data)
        X1, y1 = self.prepare_features(game1_processed, 'game1')
        results1 = self.train_model(X1, y1, balance_method=balance_method)
        importance1 = self.evaluate_model(results1)

        # Game 2 with 12 features
        print("\n=== GAME 2 ANALYSIS ===")
        predictor2 = GameChurnPredictor()
        game2_processed = predictor2.define_churn(game2_data)
        X2, y2 = predictor2.prepare_features(game2_processed, 'game2')
        results2 = predictor2.train_model(X2, y2, balance_method=balance_method)
        importance2 = predictor2.evaluate_model(results2)

        print("\n=== Separate Feature Rankings ===")
        print(f"\nGame 1 Rankings (10 features):")
        for i, (_, row) in enumerate(importance1.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:20s}: {row['importance']:.4f}")

        print(f"\nGame 2 Rankings (12 features):")
        for i, (_, row) in enumerate(importance2.iterrows(), 1):
            print(f"{i:2d}. {row['feature']:20s}: {row['importance']:.4f}")

        # Show common features comparison
        print(f"\n=== Common Features Comparison ===")
        common_features = set(importance1['feature']) & set(importance2['feature'])

        for feature in common_features:
            rank1 = importance1[importance1['feature'] == feature].index[0] + 1
            rank2 = importance2[importance2['feature'] == feature].index[0] + 1
            imp1 = importance1[importance1['feature'] == feature]['importance'].iloc[0]
            imp2 = importance2[importance2['feature'] == feature]['importance'].iloc[0]

            print(f"{feature:20s}: Game1 #{rank1:2d} ({imp1:.4f}) | Game2 #{rank2:2d} ({imp2:.4f})")

        return importance1, importance2


def main():
    predictor = GameChurnPredictor()

    # Load data
    game1_data = predictor.load_game1_data('game1_features_v2.csv')
    game2_data = predictor.load_game2_data('game2_features_v2.csv', 'user_level_aggregates.csv')

    # Compare balancing methods on one dataset first
    print("\n=== COMPARING BALANCING METHODS ON GAME 1 ===")
    game1_processed = predictor.define_churn(game1_data)
    X1, y1 = predictor.prepare_features(game1_processed, 'game1')
    comparison = predictor.compare_balancing_methods(X1, y1)

    # Use best balancing method for full analysis
    best_method = comparison['F1'].idxmax()
    print(f"\nBest balancing method based on F1 score: {best_method}")

    # Full analysis with balanced datasets
    print(f"\n=== FULL ANALYSIS WITH {best_method.upper()} BALANCING ===")
    predictor.get_separate_rankings_original_features(game1_data, game2_data, balance_method=best_method)


if __name__ == "__main__":
    main()

    
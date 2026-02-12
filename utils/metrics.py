"""Model Performance Metrics Calculator"""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, 
    f1_score, confusion_matrix, precision_score, recall_score,
    classification_report
)
import warnings
warnings.filterwarnings('ignore')


class ModelMetricsCalculator:
    """Calculate and analyze model performance metrics"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_scores=None):
        """
        Calculate comprehensive performance metrics
        
        Parameters:
        -----------
        y_true : array-like, actual labels
        y_pred : array-like, predicted labels
        y_scores : array-like, prediction scores/probabilities
        """
        metrics = {
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
        
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                metrics['roc_curve'] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist()}
                
                precision, recall, _ = precision_recall_curve(y_true, y_scores)
                metrics['pr_curve'] = {'precision': precision.tolist(), 'recall': recall.tolist()}
            except Exception as e:
                metrics['roc_auc'] = None
                metrics['roc_curve'] = None
                metrics['pr_curve'] = None
        
        return metrics
    
    @staticmethod
    def calculate_confidence_intervals(predictions, confidence=0.95):
        """Calculate confidence intervals for predictions"""
        n = len(predictions)
        mean = np.mean(predictions)
        std_err = np.std(predictions) / np.sqrt(n)
        margin = 1.96 * std_err if confidence == 0.95 else 2.576 * std_err
        
        return {
            'mean': float(mean),
            'lower_bound': float(mean - margin),
            'upper_bound': float(mean + margin),
            'margin_of_error': float(margin)
        }
    
    @staticmethod
    def get_feature_statistics(X, features):
        """Get statistics for each feature"""
        stats = {}
        for feature in features:
            stats[feature] = {
                'mean': float(X[feature].mean()),
                'std': float(X[feature].std()),
                'min': float(X[feature].min()),
                'max': float(X[feature].max()),
                'median': float(X[feature].median()),
                '25_percentile': float(X[feature].quantile(0.25)),
                '75_percentile': float(X[feature].quantile(0.75))
            }
        return stats

"""Data Quality Validation"""
import pandas as pd
import numpy as np


class DataQualityValidator:
    """Validate and profile data quality"""
    
    @staticmethod
    def assess_data_quality(df):
        """
        Comprehensive data quality assessment
        
        Returns dict with quality metrics
        """
        assessment = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_data': {},
            'data_types': {},
            'duplicates': len(df[df.duplicated()]),
            'quality_score': 0.0,
            'issues': []
        }
        
        # Check missing data
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100 if len(df) > 0 else 0.0
            assessment['missing_data'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_pct)
            }
            if missing_count > 0:
                assessment['issues'].append(f"Column '{col}' has {missing_pct:.1f}% missing values")
        
        # Check data types
        for col in df.columns:
            assessment['data_types'][col] = str(df[col].dtype)
        
        # Check duplicates
        if assessment['duplicates'] > 0:
            assessment['issues'].append(f"{assessment['duplicates']} duplicate rows found")
        
        # Calculate quality score (0-100)
        max_issues = len(df.columns) + 2  # max possible issues
        quality_score = 100 * (1 - (len(assessment['issues']) / max_issues))
        assessment['quality_score'] = max(0.0, float(quality_score))
        
        return assessment
    
    @staticmethod
    def validate_features(df, required_features):
        """Validate that required features exist and are numeric"""
        issues = []
        
        # Check for missing columns
        missing = [f for f in required_features if f not in df.columns]
        if missing:
            issues.append(f"Missing columns: {', '.join(missing)}")
        
        # Check data types
        for col in required_features:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    issues.append(f"Column '{col}' is not numeric")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues
        }
    
    @staticmethod
    def detect_outliers(df, features, method='iqr', threshold=1.5):
        """Detect outliers using IQR or Z-score"""
        outliers = pd.DataFrame()
        
        for col in features:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                col_outliers = df[(df[col] < lower) | (df[col] > upper)]
            else:  # z-score
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = df[z_scores > threshold]
            
            outliers = pd.concat([outliers, col_outliers])
        
        return outliers.drop_duplicates()
    
    @staticmethod
    def get_data_profile(df):
        """Generate detailed data profile"""
        profile = {
            'shape': {'rows': len(df), 'columns': len(df.columns)},
            'columns': {}
        }
        
        for col in df.columns:
            profile['columns'][col] = {
                'type': str(df[col].dtype),
                'non_null': int(df[col].notna().sum()),
                'null': int(df[col].isnull().sum()),
                'unique': int(df[col].nunique()),
                'memory': float(df[col].memory_usage(deep=True) / 1024)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                profile['columns'][col].update({
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'median': float(df[col].median()),
                    'std': float(df[col].std())
                })
        
        return profile

#!/usr/bin/env python3
"""
Automation Script for Loan Approval Data Preprocessing
MSML Project - Kriteria 1 (Skilled/Advanced)

This script automates the preprocessing pipeline for loan approval classification data.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoanDataPreprocessor:
    """
    Automated preprocessing pipeline for loan approval data
    """
    
    def __init__(self, target_column: str = 'loan_status'):
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load dataset from CSV file"""
        logger.info(f"Loading data from {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        df_processed = df.copy()
        missing_before = df_processed.isnull().sum().sum()
        
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if df_processed[col].dtype in ['int64', 'float64']:
                    # Numerical: fill with median
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
                    logger.info(f"Filled {col} (numerical) with median")
                else:
                    # Categorical: fill with mode
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
                    logger.info(f"Filled {col} (categorical) with mode")
        
        missing_after = df_processed.isnull().sum().sum()
        logger.info(f"Missing values: {missing_before} -> {missing_after}")
        
        return df_processed
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        logger.info("Removing duplicates...")
        
        duplicates_before = df.duplicated().sum()
        df_processed = df.drop_duplicates()
        duplicates_after = df_processed.duplicated().sum()
        
        logger.info(f"Duplicates: {duplicates_before} -> {duplicates_after}")
        logger.info(f"Shape after removing duplicates: {df_processed.shape}")
        
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method (capping)"""
        logger.info("Handling outliers...")
        
        df_processed = df.copy()
        numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != self.target_column]
        
        for col in numerical_cols:
            Q1 = df_processed[col].quantile(0.25)
            Q3 = df_processed[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = ((df_processed[col] < lower_bound) | 
                             (df_processed[col] > upper_bound)).sum()
            
            # Cap outliers
            df_processed[col] = np.where(df_processed[col] < lower_bound, 
                                       lower_bound, df_processed[col])
            df_processed[col] = np.where(df_processed[col] > upper_bound, 
                                       upper_bound, df_processed[col])
            
            if outliers_before > 0:
                logger.info(f"{col}: {outliers_before} outliers capped")
        
        return df_processed
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using Label Encoder"""
        logger.info("Encoding categorical features...")
        
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != self.target_column]
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
            logger.info(f"Encoded {col}: {len(le.classes_)} categories")
        
        return df_processed
    
    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler"""
        logger.info("Scaling features...")
        
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), 
            columns=X.columns, 
            index=X.index
        )
        
        self.feature_names = X.columns.tolist()
        logger.info(f"Features scaled. Shape: {X_scaled.shape}")
        
        return X_scaled
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into train and test sets"""
        logger.info(f"Splitting data (test_size={test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
                           y_train: pd.Series, y_test: pd.Series,
                           output_dir: str = "loan_data_preprocessing") -> None:
        """Save processed data and preprocessing objects"""
        logger.info(f"Saving processed data to {output_dir}...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data splits
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        pd.DataFrame(y_train).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        pd.DataFrame(y_test).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        
        # Save preprocessing objects
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(output_dir, 'label_encoders.pkl'))
        
        # Save metadata
        metadata = {
            'target_column': self.target_column,
            'feature_names': self.feature_names,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'categorical_encoders': list(self.label_encoders.keys())
        }
        
        import json
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("All files saved successfully!")
    
    def preprocess_pipeline(self, input_file: str, output_dir: str = "loan_data_preprocessing",
                           test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        
        Args:
            input_file: Path to raw CSV file
            output_dir: Directory to save processed data
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with processing results and metrics
        """
        logger.info("Starting preprocessing pipeline...")
        
        # Step 1: Load data
        df = self.load_data(input_file)
        original_shape = df.shape
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 4: Handle outliers
        df = self.handle_outliers(df)
        
        # Step 5: Encode categorical features
        df = self.encode_categorical_features(df)
        
        # Step 6: Separate features and target
        if self.target_column in df.columns:
            X = df.drop(self.target_column, axis=1)
            y = df[self.target_column]
        else:
            logger.warning(f"Target column '{self.target_column}' not found. Using last column as target.")
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
        
        # Step 7: Scale features
        X_scaled = self.scale_features(X)
        
        # Step 8: Split data
        X_train, X_test, y_train, y_test = self.split_data(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        # Step 9: Save processed data
        self.save_processed_data(X_train, X_test, y_train, y_test, output_dir)
        
        # Summary
        results = {
            'original_shape': original_shape,
            'final_shape': df.shape,
            'train_shape': X_train.shape,
            'test_shape': X_test.shape,
            'features': self.feature_names,
            'categorical_columns': list(self.label_encoders.keys()),
            'target_distribution': {
                'train': y_train.value_counts().to_dict(),
                'test': y_test.value_counts().to_dict()
            }
        }
        
        logger.info("Preprocessing pipeline completed successfully!")
        logger.info(f"Original shape: {original_shape} -> Final shape: {df.shape}")
        
        return results

def main():
    """Main function for standalone execution"""
    # Configuration
    INPUT_FILE = "../loan_data_raw/loan_data.csv"  # Adjust path as needed
    OUTPUT_DIR = "loan_data_preprocessing"
    TARGET_COLUMN = "loan_status"  # Adjust to your target column name
    
    # Initialize preprocessor
    preprocessor = LoanDataPreprocessor(target_column=TARGET_COLUMN)
    
    try:
        # Run preprocessing pipeline
        results = preprocessor.preprocess_pipeline(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            test_size=0.2,
            random_state=42
        )
        
        # Print results
        print("\n" + "="*50)
        print("PREPROCESSING RESULTS")
        print("="*50)
        print(f"Original shape: {results['original_shape']}")
        print(f"Final shape: {results['final_shape']}")
        print(f"Train shape: {results['train_shape']}")
        print(f"Test shape: {results['test_shape']}")
        print(f"Number of features: {len(results['features'])}")
        print(f"Categorical columns encoded: {len(results['categorical_columns'])}")
        print("\nTarget distribution (train):", results['target_distribution']['train'])
        print("Target distribution (test):", results['target_distribution']['test'])
        
        return True
        
    except Exception as e:
        logger.error(f"Error in preprocessing pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Preprocessing completed successfully!")
    else:
        print("\n❌ Preprocessing failed!")
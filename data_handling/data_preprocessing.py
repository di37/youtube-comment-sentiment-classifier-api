import os, sys
from os.path import dirname as up

# Add parent directory to Python path to allow imports from utilities
sys.path.append(os.path.abspath(os.path.join(up(__file__), os.pardir)))

import numpy as np
import pandas as pd
import re
import nltk
import string
import unicodedata
import html
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import logging

# logging configuration
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Download required NLTK data
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')  # For better lemmatization

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment.
    
    This function cleans and normalizes a single comment by:
    - Converting to lowercase
    - Removing URLs
    - Removing special characters and emojis
    - Removing stopwords (except sentiment-critical ones)
    - Lemmatizing words
    
    Args:
        comment (str): Raw comment text
        
    Returns:
        str: Cleaned and preprocessed comment
    """
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove URLs (http, https, www links)
        comment = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', comment)
        comment = re.sub(r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),])+', '', comment)

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-English characters (keeping only English letters, digits, and basic punctuation)
        # This handles emojis, special symbols, and characters from other languages
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    
    except Exception as e:
        logger.error(f"Error in preprocessing comment: {e}")
        return comment


def process_comment_for_api(comment):
    """Process a single comment and extract all features needed for model prediction.
    
    This function is designed for API usage where a single comment needs to be processed.
    It extracts features from both the original and cleaned text.
    
    Args:
        comment (str): Raw comment text from the user
        
    Returns:
        dict: Dictionary containing:
            - 'clean_comment': Preprocessed comment text
            - 'word_count': Number of words in original comment
            - 'num_stop_words': Number of stopwords in original comment
            - 'num_chars': Number of characters in original comment
            - 'num_chars_cleaned': Number of characters in cleaned comment
            
    Example:
        >>> features = process_comment_for_api("This is a great video! 😊")
        >>> print(features)
        {
            'clean_comment': 'great video',
            'word_count': 6,
            'num_stop_words': 3,
            'num_chars': 25,
            'num_chars_cleaned': 11
        }
    """
    try:
        # Validate input
        if not isinstance(comment, str):
            raise ValueError("Comment must be a string")
        
        if not comment or comment.strip() == '':
            raise ValueError("Comment cannot be empty")
        
        # Extract features from original comment (before preprocessing)
        original_comment = comment.strip()
        word_count = len(original_comment.split())
        
        # Count stopwords in original comment
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        num_stop_words = len([word for word in original_comment.split() if word in stop_words])
        
        num_chars = len(original_comment)
        
        # Apply preprocessing to clean the comment
        clean_comment = preprocess_comment(original_comment)
        
        # Extract features from cleaned comment
        num_chars_cleaned = len(clean_comment)
        
        # Return all features as a dictionary
        return {
            'clean_comment': clean_comment,
            'word_count': word_count,
            'num_stop_words': num_stop_words,
            'num_chars': num_chars,
            'num_chars_cleaned': num_chars_cleaned
        }
    
    except Exception as e:
        logger.error(f"Error in processing comment for API: {e}")
        raise

def feature_engineering(df, preprocess_comment) -> pd.DataFrame:
    """Apply preprocessing to the text data in the dataframe."""
    try:
        # Removing missing values
        df.dropna(inplace=True)
        
        # Removing duplicates
        df.drop_duplicates(inplace=True)
        
        # Removing rows with empty strings
        df = df[df['Comment'].str.strip() != '']
        
        # Features from original text (before preprocessing)
        df['word_count'] = df['Comment'].apply(lambda x: len(x.split()))
        
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        df['num_stop_words'] = df['Comment'].apply(lambda x: len([word for word in x.split() if word in stop_words]))
        
        df['num_chars'] = df['Comment'].apply(len)

        # Applying the preprocessing to the comments
        df['clean_comment'] = df['Comment'].apply(preprocess_comment)
        df.drop(columns=['Comment'], inplace=True)

        # Remove rows with empty comment
        df = df[~(df['clean_comment'].str.strip() == '')]
        
        df['num_chars_cleaned'] = df['clean_comment'].apply(len)

        df['category'] = df['Sentiment'].map({'positive': 1, 'neutral': 0, 'negative': 2})
        df.drop(columns=['Sentiment'], inplace=True)

        logger.debug('Feature engineering completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

def split_data(df, test_size=0.2, val_size=0.1, random_state=42, stratify_column=None):
    """
    Split the dataframe into train, validation, and test sets.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe to split
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split (0.0 to 1.0)
    val_size : float, default=0.1
        Proportion of the dataset to include in the validation split (0.0 to 1.0)
    random_state : int, default=42
        Random state for reproducibility
    stratify_column : str, optional
        Column name to use for stratified splitting (e.g., 'category' for classification tasks)
        If None, no stratification is applied
    
    Returns:
    --------
    tuple : (train_data, val_data, test_data)
        Three dataframes containing the train, validation, and test splits
    
    Example:
    --------
    train, val, test = split_data(df, test_size=0.2, val_size=0.1, stratify_column='category')
    """
    try:
        # Validate split sizes
        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")
        if not (0 < val_size < 1):
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        if test_size + val_size >= 1:
            raise ValueError(f"test_size + val_size must be less than 1, got {test_size + val_size}")
        
        logger.debug(f"Starting data split with test_size={test_size}, val_size={val_size}, random_state={random_state}")
        
        # Prepare stratification parameter
        stratify_param = df[stratify_column] if stratify_column and stratify_column in df.columns else None
        
        if stratify_column and stratify_column not in df.columns:
            logger.warning(f"Column '{stratify_column}' not found in dataframe. Proceeding without stratification.")
        
        # First split: separate test set from train+val
        train_val_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
        
        # Calculate validation size relative to train+val set
        # val_size_adjusted ensures we get the correct proportion of the original dataset
        val_size_adjusted = val_size / (1 - test_size)
        
        # Prepare stratification for second split
        stratify_param_val = train_val_data[stratify_column] if stratify_column and stratify_column in train_val_data.columns else None
        
        # Second split: separate validation set from train
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_param_val
        )
        
        # Log the split results
        logger.debug(f"Data split completed successfully:")
        logger.debug(f"  - Train set: {len(train_data)} samples ({len(train_data)/len(df)*100:.2f}%)")
        logger.debug(f"  - Validation set: {len(val_data)} samples ({len(val_data)/len(df)*100:.2f}%)")
        logger.debug(f"  - Test set: {len(test_data)} samples ({len(test_data)/len(df)*100:.2f}%)")
        
        return train_data, val_data, test_data
    
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise

def main():
    from utilities import load_data, save_data
    from utilities import RAW_DATA_PATH, INTERIM_DATA_PATH
    
    print(">>> Stage 2: Starting Data Preprocessing pipeline...")
    # 1. Loading the data
    train_data = load_data(os.path.join(RAW_DATA_PATH, "train.csv"))
    val_data = load_data(os.path.join(RAW_DATA_PATH, "val.csv"))
    test_data = load_data(os.path.join(RAW_DATA_PATH, "test.csv"))
    
    # 2. Preprocess the dataset
    train_df = feature_engineering(train_data, preprocess_comment)
    val_df = feature_engineering(val_data, preprocess_comment)
    test_df = feature_engineering(test_data, preprocess_comment)

    print(train_df.head())
    print(val_df.head())
    print(test_df.head())

    # 3. Save the dataset
    save_data(train_df, val_df, test_df, data_path=INTERIM_DATA_PATH)

    print(">>> Stage 2: Data Preprocessing pipeline completed successfully...")

if __name__ == "__main__":
    main()
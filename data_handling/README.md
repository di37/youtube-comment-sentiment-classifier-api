# Data Handling Module

This module contains all the functions related to data ingestion, preprocessing, and splitting for the YouTube Sentiment Insights project.

## Table of Contents
- [Module Overview](#module-overview)
- [Data Ingestion](#data-ingestion)
- [Data Preprocessing](#data-preprocessing)
- [Understanding Stratification](#understanding-stratification)
- [Usage Examples](#usage-examples)

---

## Module Overview

The `data_handling` module consists of two main files:

1. **`data_ingestion.py`** - Handles downloading and loading datasets from Kaggle
2. **`data_preprocessing.py`** - Handles text preprocessing, feature engineering, and data splitting

---

## Data Ingestion

### `download_and_copy_dataset()`

Downloads a dataset from Kaggle and copies it to a local directory.

**Parameters:**
- `dataset_name` (str): Kaggle dataset identifier (default: `"atifaliak/youtube-comments-dataset"`)
- `raw_data_path` (str): Local directory to store the dataset (default: `"data/raw"`)

**Returns:**
- `str`: Path to the CSV file, or `None` if not found or error occurred

**Usage:**
```python
from data_handling.data_ingestion import download_and_copy_dataset

# Download default YouTube comments dataset
csv_path = download_and_copy_dataset()

# Download custom dataset
csv_path = download_and_copy_dataset(
    dataset_name="username/dataset-name",
    raw_data_path="data/custom"
)
```

---

## Data Preprocessing

### `preprocess_comment()`

Applies comprehensive text preprocessing transformations to a single comment.

**Processing Steps:**
1. Converts text to lowercase
2. Removes trailing and leading whitespaces
3. Removes URLs (http, https, www links)
4. Removes newline characters
5. Removes non-English characters (emojis, special symbols)
6. Removes stopwords (except sentiment-important words like 'not', 'but', 'however', 'no', 'yet')
7. Lemmatizes words to their base form

**Parameters:**
- `comment` (str): The text comment to preprocess

**Returns:**
- `str`: Preprocessed comment

**Usage:**
```python
from data_handling.data_preprocessing import preprocess_comment

text = "I absolutely LOVE this video!!! 😍 https://example.com"
cleaned = preprocess_comment(text)
# Output: "absolutely love video"
```

---

### `feature_engineering()`

Applies preprocessing to the entire dataframe and creates additional text features.

**Features Created:**
- `word_count` - Number of words in the original comment
- `num_stop_words` - Number of stopwords in the original comment
- `num_chars` - Number of characters in the original comment
- `num_chars_cleaned` - Number of characters after preprocessing

**Parameters:**
- `df` (pd.DataFrame): Input dataframe with a `clean_comment` column
- `preprocess_comment` (function): The preprocessing function to apply

**Returns:**
- `pd.DataFrame`: Dataframe with preprocessed text and additional features

**Usage:**
```python
from data_handling.data_preprocessing import feature_engineering, preprocess_comment

df = feature_engineering(df, preprocess_comment)
```

---

### `split_data()`

Splits the dataframe into train, validation, and test sets with optional stratification.

**Parameters:**
- `df` (pd.DataFrame): The input dataframe to split
- `test_size` (float): Proportion for test split (default: 0.2 = 20%)
- `val_size` (float): Proportion for validation split (default: 0.1 = 10%)
- `random_state` (int): Random seed for reproducibility (default: 42)
- `stratify_column` (str, optional): Column name for stratified splitting (e.g., 'category')

**Returns:**
- `tuple`: (train_data, val_data, test_data) - Three dataframes

**Usage:**
```python
from data_handling.data_preprocessing import split_data

# Basic usage (70% train, 10% val, 20% test)
train, val, test = split_data(df)

# Custom split ratios (70% train, 15% val, 15% test)
train, val, test = split_data(df, test_size=0.15, val_size=0.15)

# With stratification for classification tasks
train, val, test = split_data(
    df, 
    test_size=0.2, 
    val_size=0.1, 
    stratify_column='sentiment'
)
```

---

### `save_data()`

Saves the processed train, validation, and test datasets to CSV files.

**Parameters:**
- `train_data` (pd.DataFrame): Training dataset
- `val_data` (pd.DataFrame): Validation dataset
- `test_data` (pd.DataFrame): Test dataset
- `data_path` (str): Base directory path (files saved in `{data_path}/interim/`)

**Output Files:**
- `train_processed.csv`
- `val_processed.csv`
- `test_processed.csv`

**Usage:**
```python
from data_handling.data_preprocessing import save_data

save_data(train_data, val_data, test_data, data_path="data")
# Files saved to: data/interim/
```

---

## Understanding Stratification

### What is Stratification?

**Stratification** ensures that the proportion of classes (categories) in your target variable is maintained consistently across all data splits (train, validation, and test sets).

### Why is Stratification Important?

Without stratification, random splitting might create imbalanced splits where some classes are:
- Over-represented in one split and under-represented in another
- Completely missing from certain splits
- Unevenly distributed across splits

This can lead to:
- Poor model generalization
- Biased performance metrics
- Training on unrepresentative data

### Example Without Stratification ❌

**Original Dataset**: 1000 comments
- 700 Positive (70%)
- 200 Negative (20%)
- 100 Neutral (10%)

**Possible Random Split**:
- **Train**: 600 Positive, 100 Negative, 100 Neutral
  - Proportions: 75% / 12.5% / 12.5%
- **Test**: 100 Positive, 100 Negative, 0 Neutral
  - Proportions: 50% / 50% / 0%

**Problems**:
- Test set has no Neutral examples (model can't evaluate Neutral predictions)
- Class proportions differ significantly between train and test
- Model trained on 75% positive might not generalize well to 50% positive test set

### Example With Stratification ✅

**Original Dataset**: 1000 comments
- 700 Positive (70%)
- 200 Negative (20%)
- 100 Neutral (10%)

**Stratified Split**:
- **Train (70%)**: 490 Positive, 140 Negative, 70 Neutral
  - Proportions: 70% / 20% / 10% ✓
- **Validation (10%)**: 70 Positive, 20 Negative, 10 Neutral
  - Proportions: 70% / 20% / 10% ✓
- **Test (20%)**: 140 Positive, 40 Negative, 20 Neutral
  - Proportions: 70% / 20% / 10% ✓

**Benefits**:
- All splits have identical class distributions
- Each split is representative of the full dataset
- Fair evaluation across all classes
- Better model generalization

### When to Use Stratification?

#### ✅ **Use Stratification When:**
- Working on **classification tasks** (sentiment analysis, spam detection, etc.)
- Target variable has **categorical classes**
- Dealing with **imbalanced datasets** (e.g., 90% positive, 10% negative)
- You want **fair representation** of all classes in each split
- You need **reliable evaluation metrics** that reflect real-world distribution

#### ❌ **Don't Use Stratification When:**
- Working on **regression tasks** (predicting continuous values like price, temperature)
- Target variable is **continuous** (not categorical)
- You have **very few samples** per class (might cause split errors)
- Your dataset is **perfectly balanced** and large (stratification won't hurt, but less critical)

### How Stratification Works in `split_data()`

The function performs stratification in two steps:

**Step 1: Split into (Train + Val) and Test**
```python
# Maintains class proportions when separating test set
train_val_data, test_data = train_test_split(
    df,
    test_size=0.2,
    stratify=df['sentiment']  # Ensures test set has same class ratios
)
```

**Step 2: Split (Train + Val) into Train and Val**
```python
# Maintains class proportions when separating validation set
train_data, val_data = train_test_split(
    train_val_data,
    test_size=adjusted_val_size,
    stratify=train_val_data['sentiment']  # Ensures val set has same class ratios
)
```

### Stratification in Practice

```python
# Example: YouTube sentiment dataset with imbalanced classes
df = pd.read_csv('youtube_comments.csv')

# Check class distribution
print(df['sentiment'].value_counts(normalize=True))
# Output:
#   positive    0.65  (65%)
#   negative    0.25  (25%)
#   neutral     0.10  (10%)

# Split WITHOUT stratification (not recommended)
train, val, test = split_data(df)
# Risk: Random chance might give you test set with 80% positive, 20% negative, 0% neutral

# Split WITH stratification (recommended)
train, val, test = split_data(df, stratify_column='sentiment')
# Guarantee: All splits maintain 65% positive, 25% negative, 10% neutral
```

### Verification Example

After splitting with stratification, you can verify the distributions:

```python
# Split data with stratification
train, val, test = split_data(df, stratify_column='sentiment')

# Verify distributions
print("Train distribution:")
print(train['sentiment'].value_counts(normalize=True))

print("\nValidation distribution:")
print(val['sentiment'].value_counts(normalize=True))

print("\nTest distribution:")
print(test['sentiment'].value_counts(normalize=True))

# All three should show similar proportions:
#   positive    ~0.65
#   negative    ~0.25
#   neutral     ~0.10
```

---

## Usage Examples

### Complete Pipeline Example

```python
from data_handling.data_ingestion import download_and_copy_dataset
from data_handling.data_preprocessing import (
    preprocess_comment,
    feature_engineering,
    split_data,
    save_data
)
import pandas as pd

# Step 1: Download dataset from Kaggle
csv_path = download_and_copy_dataset()

# Step 2: Load the dataset
df = pd.read_csv(csv_path)

# Step 3: Feature engineering and preprocessing
df = feature_engineering(df, preprocess_comment)

# Step 4: Split data with stratification
train, val, test = split_data(
    df,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    stratify_column='category'  # Use your target column name
)

# Step 5: Save processed datasets
save_data(train, val, test, data_path='data')

print(f"Train set: {len(train)} samples")
print(f"Validation set: {len(val)} samples")
print(f"Test set: {len(test)} samples")
```

### Custom Preprocessing Pipeline

```python
from data_handling.data_preprocessing import preprocess_comment

# Process individual comments
comments = [
    "This is AMAZING!!! 😍😍😍",
    "I don't like this at all 😞",
    "Pretty good video, thanks!"
]

processed_comments = [preprocess_comment(c) for c in comments]
print(processed_comments)
# Output: ['amazing', 'dont like', 'pretty good video thank']
```

---

## Logging

Both modules include comprehensive logging:

- **Console logs**: INFO and DEBUG level messages
- **File logs**: ERROR level messages saved to:
  - `errors.log` (data_ingestion)
  - `preprocessing_errors.log` (data_preprocessing)

The logs track:
- Dataset download progress
- Preprocessing steps
- Data split statistics
- Any errors or warnings

---

## Dependencies

Required packages:
```
pandas
numpy
scikit-learn
nltk
kagglehub
pyyaml
```

Required NLTK data (automatically downloaded):
- `wordnet` - For lemmatization
- `stopwords` - For stopword removal
- `omw-1.4` - For better lemmatization

---

## Best Practices

1. **Always use stratification** for classification tasks
2. **Set a fixed random_state** for reproducibility
3. **Verify class distributions** after splitting
4. **Keep validation and test sets separate** - never train on them
5. **Apply the same preprocessing** to all splits
6. **Log all operations** for debugging and tracking

---

## Troubleshooting

### Issue: "Column not found" error in stratification
**Solution**: Ensure the `stratify_column` name matches exactly with your dataframe column name.

### Issue: Stratification fails with small datasets
**Solution**: If you have very few samples per class, either:
- Don't use stratification
- Increase dataset size
- Combine rare classes

### Issue: Preprocessing removes too much text
**Solution**: Adjust the stopwords list or regex patterns in `preprocess_comment()` to retain more content.

---

## Contributing

When adding new functions to this module:
1. Include comprehensive docstrings
2. Add error handling with proper logging
3. Update this README with usage examples
4. Add unit tests in the `tests/` directory

---

## License

This module is part of the YouTube Sentiment Insights project.


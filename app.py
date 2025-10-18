"""
YouTube Sentiment Analysis API

This API provides sentiment analysis for YouTube comments using a trained LightGBM model.
It processes comments and returns sentiment predictions (positive, neutral, or negative).
"""

import os
import sys
import pickle
import logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict
import mlflow
import mlflow.pyfunc
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set AWS credentials for MLflow S3 access
os.environ['AWS_ACCESS_KEY_ID'] = os.getenv('AWS_ACCESS_KEY_ID')
os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv('AWS_SECRET_ACCESS_KEY')
os.environ['AWS_DEFAULT_REGION'] = os.getenv('AWS_DEFAULT_REGION')

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the preprocessing function
from data_handling.data_preprocessing import process_comment_for_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YouTube Sentiment Analysis API",
    description="Analyze sentiment of YouTube comments",
    version="1.0.0"
)

# Global variables for models and vectorizer
local_model = None  # Model from local pickle file
mlflow_model = None  # Model from MLflow registry
vectorizer = None

# Sentiment mapping
# Model output: 0=neutral, 1=positive, 2=negative
# API output: 0=neutral, 1=positive, -1=negative
SENTIMENT_MAP = {
    0: 0,   # neutral
    1: 1,   # positive
    2: -1   # negative
}

SENTIMENT_LABELS = {
    0: "neutral",
    1: "positive",
    -1: "negative"
}


class CommentRequest(BaseModel):
    """Request model for comment sentiment analysis."""
    comment: str = Field(
        ...,
        description="The YouTube comment to analyze",
        example="This is an amazing video! I learned so much!"
    )


class SentimentResponse(BaseModel):
    """Response model for sentiment prediction."""
    comment: str = Field(description="Original comment")
    sentiment: int = Field(description="Predicted sentiment (1=positive, 0=neutral, -1=negative)")


class BatchCommentRequest(BaseModel):
    """Request model for batch comment sentiment analysis."""
    comment: list[str] = Field(
        ...,
        description="List of YouTube comments to analyze",
        example=["This is an amazing video!", "Terrible content"]
    )


def load_models_and_vectorizer():
    """Load both local and MLflow models along with the vectorizer.
    
    This function loads:
    - Local model from pickle file (lgbm_model.pkl)
    - MLflow model from Model Registry (staging alias)
    - TF-IDF vectorizer from local pickle file
    """
    global local_model, mlflow_model, vectorizer
    
    # Load TF-IDF vectorizer (shared by both models)
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info("✓ TF-IDF vectorizer loaded from models/tfidf_vectorizer.pkl")
    except Exception as e:
        logger.error(f"Error loading vectorizer: {e}")
        raise
    
    # Load local model
    try:
        with open('models/lgbm_model.pkl', 'rb') as f:
            local_model = pickle.load(f)
        logger.info("✓ Local model loaded from models/lgbm_model.pkl")
    except Exception as e:
        logger.error(f"Error loading local model: {e}")
        logger.warning("Local model endpoints will not be available")
    
    # Load MLflow model
    try:
        logger.info("Loading model from MLflow Model Registry...")
        mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://3.29.129.159:5000')
        logger.info(f"Using MLflow tracking URI: {mlflow_tracking_uri}")
        
        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Set S3 endpoint URL if using AWS
        aws_endpoint = os.getenv('AWS_ENDPOINT_URL')
        if aws_endpoint:
            os.environ['MLFLOW_S3_ENDPOINT_URL'] = aws_endpoint
        
        # Test MLflow connection
        try:
            mlflow.search_experiments()
            logger.info("Successfully connected to MLflow server")
        except Exception as conn_err:
            logger.error(f"Failed to connect to MLflow server: {conn_err}")
            raise
        
        # Load the model with staging alias using sklearn flavor
        model_name = "yt_chrome_plugin_model"
        model_version = "latest"
        try:
            mlflow_model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
            logger.info("✓ MLflow model loaded from Model Registry (staging)")
        except Exception as model_err:
            logger.error(f"Failed to load model {model_name}: {model_err}")
            # Fallback to local model
            logger.info("Attempting to use local model only...")
            mlflow_model = None
            
    except Exception as e:
        logger.error(f"Error in MLflow setup: {e}")
        logger.warning("MLflow model endpoints will not be available")


@app.on_event("startup")
async def startup_event():
    """Load models and vectorizer when the API starts."""
    logger.info("Starting YouTube Sentiment Analysis API...")
    load_models_and_vectorizer()
    logger.info("API is ready to accept requests!")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "YouTube Sentiment Analysis API",
        "version": "1.0.0",
        "models": {
            "local": "Model from local disk (models/lgbm_model.pkl)",
            "mlflow": "Model from MLflow Model Registry (staging)"
        },
        "endpoints": {
            "/predict": "POST - Single prediction (local model)",
            "/batch_predict": "POST - Batch predictions (local model)",
            "/predict_mlflow": "POST - Single prediction (MLflow model)",
            "/batch_predict_mlflow": "POST - Batch predictions (MLflow model)",
            "/health": "GET - Check API health status",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "local_model_loaded": local_model is not None,
        "mlflow_model_loaded": mlflow_model is not None,
        "vectorizer_loaded": vectorizer is not None
    }


def make_prediction(comment_text: str, model_to_use):
    """Helper function to make a sentiment prediction.
    
    Args:
        comment_text: The comment to analyze
        model_to_use: The model to use for prediction (local_model or mlflow_model)
        
    Returns:
        Sentiment value (1, 0, or -1)
    """
    # Process the comment and extract features
    features = process_comment_for_api(comment_text)
    
    # Check if cleaned comment is empty
    if not features['clean_comment'] or features['clean_comment'].strip() == '':
        return 0  # neutral for empty comments
    
    # Transform the cleaned comment using TF-IDF vectorizer
    tfidf_features = vectorizer.transform([features['clean_comment']]).toarray()
    
    # Prepare numerical features in the same order as during training
    numerical_features = np.array([[
        features['word_count'],
        features['num_stop_words'],
        features['num_chars'],
        features['num_chars_cleaned']
    ]])
    
    # Combine TF-IDF features with numerical features
    X = np.hstack([tfidf_features, numerical_features])
    
    # Make prediction
    prediction = model_to_use.predict(X)[0]
    
    # Map prediction to sentiment (1=positive, 0=neutral, -1=negative)
    sentiment = SENTIMENT_MAP.get(int(prediction), 0)
    sentiment_label = SENTIMENT_LABELS.get(sentiment, "unknown")
    
    logger.info(f"Predicted sentiment: {sentiment_label} ({sentiment})")
    
    return sentiment


@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: CommentRequest):
    """
    Predict sentiment for a YouTube comment using LOCAL model.
    
    This endpoint uses the model from local disk (models/lgbm_model.pkl).
    
    Args:
        request: CommentRequest containing the comment to analyze
        
    Returns:
        SentimentResponse with prediction results
    """
    try:
        # Validate that model and vectorizer are loaded
        if local_model is None or vectorizer is None:
            raise HTTPException(
                status_code=503,
                detail="Local model or vectorizer not loaded. Please check server logs."
            )
        
        sentiment = make_prediction(request.comment, local_model)
        
        return SentimentResponse(
            comment=request.comment,
            sentiment=sentiment
        )
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )


@app.post("/batch_predict")
async def batch_predict(request: BatchCommentRequest):
    """
    Predict sentiment for multiple comments using LOCAL model.
    
    This endpoint uses the model from local disk (models/lgbm_model.pkl).
    
    Request format:
    {
        "comment": ["This is great!", "Very bad video"]
    }
    
    Response format:
    [
        {"comment": "This is great!", "sentiment": 1},
        {"comment": "Very bad video", "sentiment": -1}
    ]
    
    Args:
        request: BatchCommentRequest containing list of comments
        
    Returns:
        List of SentimentResponse objects
    """
    try:
        if local_model is None or vectorizer is None:
            raise HTTPException(
                status_code=503,
                detail="Local model or vectorizer not loaded."
            )
        
        results = []
        for comment_text in request.comment:
            sentiment = make_prediction(comment_text, local_model)
            results.append(SentimentResponse(comment=comment_text, sentiment=sentiment))
        
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during batch prediction: {str(e)}"
        )


@app.post("/predict_mlflow", response_model=SentimentResponse)
async def predict_sentiment_mlflow(request: CommentRequest):
    """
    Predict sentiment for a YouTube comment using MLFLOW model.
    
    This endpoint uses the model from MLflow Model Registry (staging).
    
    Args:
        request: CommentRequest containing the comment to analyze
        
    Returns:
        SentimentResponse with prediction results
    """
    try:
        # Validate that model and vectorizer are loaded
        if mlflow_model is None or vectorizer is None:
            raise HTTPException(
                status_code=503,
                detail="MLflow model or vectorizer not loaded. Please check server logs."
            )
        
        sentiment = make_prediction(request.comment, mlflow_model)
        
        return SentimentResponse(
            comment=request.comment,
            sentiment=sentiment
        )
        
    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )


@app.post("/batch_predict_mlflow")
async def batch_predict_mlflow(request: BatchCommentRequest):
    """
    Predict sentiment for multiple comments using MLFLOW model.
    
    This endpoint uses the model from MLflow Model Registry (staging).
    
    Request format:
    {
        "comment": ["This is great!", "Very bad video"]
    }
    
    Response format:
    [
        {"comment": "This is great!", "sentiment": 1},
        {"comment": "Very bad video", "sentiment": -1}
    ]
    
    Args:
        request: BatchCommentRequest containing list of comments
        
    Returns:
        List of SentimentResponse objects
    """
    try:
        if mlflow_model is None or vectorizer is None:
            raise HTTPException(
                status_code=503,
                detail="MLflow model or vectorizer not loaded."
            )
        
        results = []
        for comment_text in request.comment:
            sentiment = make_prediction(comment_text, mlflow_model)
            results.append(SentimentResponse(comment=comment_text, sentiment=sentiment))
        
        return results
        
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during batch prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    # Access the API at: http://localhost:6889
    # Interactive docs at: http://localhost:6889/docs
    uvicorn.run(app, host="0.0.0.0", port=6889, log_level="info")


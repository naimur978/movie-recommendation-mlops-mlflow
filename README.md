# Movie Recommendation System with MLflow

This project implements a content-based movie recommendation system using TMDB dataset, with MLflow for experiment tracking and model management. The system is deployed as a FastAPI service.

## Live Demo

API Endpoint: https://movie-recommendation-mlops-mlflow.onrender.com

- API Documentation (Swagger UI): https://movie-recommendation-mlops-mlflow.onrender.com/docs
- Alternative Documentation (ReDoc): https://movie-recommendation-mlops-mlflow.onrender.com/redoc

## API Endpoints

- `GET /` - Welcome endpoint with API information
- `GET /health` - Health check endpoint
- `GET /movies` - List available movies (with pagination)
- `GET /recommend` - Get movie recommendations

## Features

- Content-based movie recommendation using:
  - Movie genres
  - Cast members
  - Director
  - Keywords
- MLflow integration for:
  - Experiment tracking
  - Parameter logging
  - Metric monitoring
  - Artifact management
- FastAPI implementation for:
  - RESTful API endpoints
  - Interactive documentation
  - Data validation
  - Error handling
- Text processing with:
  - CountVectorizer
  - Porter Stemming
  - Cosine Similarity

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/naimur978/movie-recommendation-mlops-mlflow.git
   cd movie-recommendation-mlops-mlflow
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scriptsctivate
   ```

3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn nltk mlflow
   ```

4. Download the TMDB dataset and place in `dataset/` folder:
   - tmdb_5000_movies.csv
   - tmdb_5000_credits.csv

## Usage

Run the recommendation system:
```bash
python movie-recommender-using-vectorization.py
```

View MLflow UI:
```bash
mlflow ui
```

## MLflow Tracking

The project tracks:
- Dataset parameters
- Model parameters (max_features, vectorizer type)
- Processing times
- Feature statistics
- Model artifacts
- Recommendation metrics

## Project Structure

```
├── dataset/
│   ├── tmdb_5000_movies.csv
│   ├── tmdb_5000_credits.csv
│   └── count_cosine_recommendations.csv
├── models/
├── artifacts/
├── app.py
├── movie-recommender-using-vectorization.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## API Usage Examples

### Get Movie Recommendations

```bash
curl "https://movie-recommendation-mlops-mlflow.onrender.com/recommend?movie_title=Avatar&num_recommendations=5"
```

### List Available Movies

```bash
curl "https://movie-recommendation-mlops-mlflow.onrender.com/movies?limit=10&skip=0"
```

### Check API Health

```bash
curl "https://movie-recommendation-mlops-mlflow.onrender.com/health"
```

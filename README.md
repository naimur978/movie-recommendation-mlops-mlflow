# Movie Recommendation System with MLflow

This project implements a movie recommendation system using content-based filtering, with MLflow for experiment tracking and model management.

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
│   └── tmdb_5000_credits.csv
├── models/
├── artifacts/
├── movie-recommender-using-vectorization.py
├── README.md
└── .gitignore
```

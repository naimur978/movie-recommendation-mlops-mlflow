import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dataset paths for testing
os.environ['TEST_MODE'] = 'true'

from movie_recommender import (
    load_data,
    convert_list_of_dict_to_names,
    preprocess_data,
    create_recommendation_model,
    recommend_movies,
    DATASET_CONFIG
)

@pytest.fixture
def sample_data():
    movies = pd.DataFrame({
        'id': [1, 2],
        'title': ['Movie 1', 'Movie 2'],
        'genres': ['[{"id": 1, "name": "Action"}]', '[{"id": 2, "name": "Drama"}]'],
        'keywords': ['[{"id": 1, "name": "hero"}]', '[{"id": 2, "name": "family"}]'],
        'overview': ['Overview 1', 'Overview 2']
    })
    
    credits = pd.DataFrame({
        'movie_id': [1, 2],
        'title': ['Movie 1', 'Movie 2'],
        'cast': ['[{"name": "Actor 1"}]', '[{"name": "Actor 2"}]'],
        'crew': ['[{"job": "Director", "name": "Director 1"}]', '[{"job": "Director", "name": "Director 2"}]']
    })
    
    return movies, credits

def test_convert_list_of_dict_to_names():
    test_input = '[{"id": 1, "name": "Action"}]'
    result = convert_list_of_dict_to_names(test_input)
    assert result == ['Action']

def test_preprocess_data(sample_data):
    movies, credits = sample_data
    cleaned_df = preprocess_data(movies, credits)
    assert 'tag' in cleaned_df.columns
    assert len(cleaned_df) == 2

def test_create_recommendation_model(sample_data):
    movies, credits = sample_data
    cleaned_df = preprocess_data(movies, credits)
    params = {
        "max_features": 1000,
        "vectorizer_type": "count",
        "stem_words": True,
        "remove_stopwords": True,
        "ngram_range": (1, 1)
    }
    vectorizer, cosine_sim, sigmoid_sim = create_recommendation_model(cleaned_df, params)
    assert cosine_sim.shape == (2, 2)
    assert sigmoid_sim.shape == (2, 2)

def test_recommend_movies(sample_data):
    movies, credits = sample_data
    cleaned_df = preprocess_data(movies, credits)
    params = {
        "max_features": 1000,
        "vectorizer_type": "count",
        "stem_words": True,
        "remove_stopwords": True,
        "ngram_range": (1, 1)
    }
    vectorizer, cosine_sim, sigmoid_sim = create_recommendation_model(cleaned_df, params)
    
    for method in ['cosine', 'sigmoid', 'hybrid']:
        recommendations = recommend_movies("Movie 1", cleaned_df, cosine_sim, sigmoid_sim, top_n=1, method=method)
        assert len(recommendations) == 1
        assert 'title' in recommendations[0]
        assert 'similarity' in recommendations[0]
        assert 'method' in recommendations[0]
        assert recommendations[0]['method'] == method

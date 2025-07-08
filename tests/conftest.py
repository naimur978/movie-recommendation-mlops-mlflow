import pytest
import os
import pandas as pd

@pytest.fixture(autouse=True)
def setup_test_env():
    """Setup test environment before each test."""
    os.environ['TEST_MODE'] = 'true'
    
    # Create test dataset directory
    os.makedirs('./dataset', exist_ok=True)
    
    # Create test datasets
    movies_data = pd.DataFrame({
        'id': [1, 2],
        'title': ['Movie 1', 'Movie 2'],
        'genres': ['[{"id": 1, "name": "Action"}]', '[{"id": 2, "name": "Drama"}]'],
        'keywords': ['[{"id": 1, "name": "hero"}]', '[{"id": 2, "name": "family"}]'],
        'overview': ['Overview 1', 'Overview 2'],
        'cast': ['[{"name": "Actor 1"}]', '[{"name": "Actor 2"}]'],
        'crew': ['[{"job": "Director", "name": "Director 1"}]', '[{"job": "Director", "name": "Director 2"}]']
    })
    
    credits_data = pd.DataFrame({
        'title': ['Movie 1', 'Movie 2'],
        'cast': ['[{"name": "Actor 1"}]', '[{"name": "Actor 2"}]'],
        'crew': ['[{"job": "Director", "name": "Director 1"}]', '[{"job": "Director", "name": "Director 2"}]']
    })
    
    movies_data.to_csv('./dataset/test_movies.csv', index=False)
    credits_data.to_csv('./dataset/test_credits.csv', index=False)
    
    yield
    
    # Cleanup
    try:
        os.remove('./dataset/test_movies.csv')
        os.remove('./dataset/test_credits.csv')
    except:
        pass

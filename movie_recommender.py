import numpy as np
import pandas as pd
import os
import ast
import time
import mlflow
import datetime
import json
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# Set up MLflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("movie-recommender")

# Create directories for models and artifacts
Path("models").mkdir(exist_ok=True)
Path("artifacts").mkdir(exist_ok=True)

# Dataset parameters
DATASET_CONFIG = {
    "movies_path": "./dataset/tmdb_5000_movies.csv",
    "credits_path": "./dataset/tmdb_5000_credits.csv",
    "movies_columns": ["id", "title", "genres", "keywords", "overview", "cast", "crew"],
    "top_cast_members": 3
}

# Modify paths for testing environment
if os.getenv('TEST_MODE') == 'true':
    DATASET_CONFIG["movies_path"] = "./dataset/test_movies.csv"
    DATASET_CONFIG["credits_path"] = "./dataset/test_credits.csv"

def load_data():
    """Load and preprocess the movie datasets."""
    movies = pd.read_csv(DATASET_CONFIG["movies_path"])
    credits = pd.read_csv(DATASET_CONFIG["credits_path"])
    return movies, credits

def convert_list_of_dict_to_names(obj, limit=None):
    """Convert a list of dictionaries to a list of names."""
    result = []
    count = 0
    for item in ast.literal_eval(obj):
        if limit and count >= limit:
            break
        if item.get('job') == 'Director':
            result.append(item['name'])
            break
        else:
            result.append(item['name'])
            count += 1
    return result

def preprocess_data(movies, credits):
    """Preprocess the movie data."""
    # Merge dataframes
    movies = movies.merge(credits, on='title')
    movies = movies[DATASET_CONFIG["movies_columns"]]
    movies.dropna(inplace=True)

    # Convert columns to lists of names
    movies["genres"] = movies["genres"].apply(lambda x: convert_list_of_dict_to_names(x))
    movies["keywords"] = movies["keywords"].apply(lambda x: convert_list_of_dict_to_names(x))
    movies["cast"] = movies["cast"].apply(lambda x: convert_list_of_dict_to_names(x, DATASET_CONFIG["top_cast_members"]))
    movies["crew"] = movies["crew"].apply(lambda x: convert_list_of_dict_to_names(x))

    # Remove spaces from values
    for col in ['cast', 'genres', 'crew', 'keywords']:
        movies[col] = movies[col].apply(lambda x: [i.replace(" ", "") for i in x])

    # Create tags
    movies['tag'] = movies['genres'] + movies["cast"] + movies['crew'] + movies['keywords']
    movies['tag'] = movies['tag'].apply(lambda x: " ".join(x))

    return movies[['id', 'title', 'tag']]

def create_recommendation_model(data, params):
    """Create and train the recommendation model."""
    cv = CountVectorizer(max_features=params["max_features"], stop_words='english')
    vectors = cv.fit_transform(data['tag']).toarray()
    
    ps = PorterStemmer()
    data['tags'] = data['tag'].apply(lambda text: " ".join([ps.stem(word) for word in text.split()]))
    
    similarity = cosine_similarity(vectors)
    return cv, similarity

def recommend_movies(movie_title, cleaned_df, similarity_matrix, top_n=5):
    """Generate movie recommendations."""
    try:
        movie_index = cleaned_df[cleaned_df['title'] == movie_title].index[0]
        distances = similarity_matrix[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
        
        recommendations = []
        for i in movies_list:
            recommendations.append({
                'title': cleaned_df.iloc[i[0]].title,
                'similarity': i[1]
            })
        return recommendations
    except IndexError:
        print(f"Movie '{movie_title}' not found in database.")
        return []

def main():
    """Main function to run the recommendation system."""
    with mlflow.start_run(run_name=f"movie_recommender_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Load and preprocess data
        start_time = time.time()
        movies, credits = load_data()
        
        # Log dataset info
        dataset_info = {
            "movies_rows": len(movies),
            "credits_rows": len(credits),
            "movies_columns": list(movies.columns),
            "credits_columns": list(credits.columns)
        }
        mlflow.log_dict(dataset_info, "dataset_info.json")
        mlflow.log_params(DATASET_CONFIG)
        
        # Model parameters
        model_params = {
            "max_features": 5000,
            "vectorizer": "CountVectorizer",
            "similarity_metric": "cosine",
            "stem_words": True,
            "remove_stopwords": True
        }
        mlflow.log_params(model_params)
        
        # Preprocess data
        cleaned_df = preprocess_data(movies, credits)
        preprocessing_time = time.time() - start_time
        
        # Create model
        model_start_time = time.time()
        vectorizer, similarity_matrix = create_recommendation_model(cleaned_df, model_params)
        model_time = time.time() - model_start_time
        
        # Log metrics
        mlflow.log_metrics({
            "preprocessing_time_seconds": preprocessing_time,
            "model_creation_time_seconds": model_time,
            "total_time_seconds": time.time() - start_time,
            "feature_count": len(vectorizer.get_feature_names_out()),
            "vocabulary_size": len(vectorizer.vocabulary_)
        })
        
        # Save model artifacts
        model_path = f"models/model_{mlflow.active_run().info.run_id}"
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        np.save(f"{model_path}/similarity_matrix.npy", similarity_matrix)
        pd.to_pickle(vectorizer, f"{model_path}/vectorizer.pkl")
        
        with open(f"{model_path}/model_info.json", "w") as f:
            json.dump({
                "feature_names": vectorizer.get_feature_names_out().tolist(),
                "movie_indices": cleaned_df['title'].to_list()
            }, f)
        
        mlflow.log_artifacts(model_path, "model")
        
        # Generate example recommendations
        example_movie = "Avatar"
        recommendations = recommend_movies(example_movie, cleaned_df, similarity_matrix)
        
        # Log recommendations
        if recommendations:
            recommendations_df = pd.DataFrame(recommendations)
            recommendations_df.to_csv("recommendations.csv", index=False)
            mlflow.log_artifact("recommendations.csv", "recommendations")
            
            for movie in recommendations:
                print(f"{movie['title']} (similarity: {movie['similarity']:.4f})")

if __name__ == "__main__":
    main()

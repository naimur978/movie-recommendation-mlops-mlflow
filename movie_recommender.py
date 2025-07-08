import numpy as np
import pandas as pd
import os
import ast
import time
import mlflow
import datetime
import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity, sigmoid_kernel
from pathlib import Path
from typing import Tuple, List, Dict, Any

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

def create_recommendation_model(data: pd.DataFrame, params: Dict[str, Any]) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Create and train the recommendation model using multiple approaches.
    
    Args:
        data: DataFrame containing movie data with tags
        params: Dictionary containing model parameters
    
    Returns:
        Tuple containing:
        - Vectorizer (CountVectorizer or TfidfVectorizer)
        - Cosine similarity matrix
        - Sigmoid kernel similarity matrix
    """
    # Apply stemming
    ps = PorterStemmer()
    data['tags'] = data['tag'].apply(lambda text: " ".join([ps.stem(word) for word in text.split()]))
    
    if params["vectorizer_type"] == "tfidf":
        # TF-IDF approach
        vectorizer = TfidfVectorizer(
            max_features=params["max_features"],
            stop_words='english',
            ngram_range=(1, 2)  # Include both unigrams and bigrams
        )
    else:
        # Count vectorizer approach
        vectorizer = CountVectorizer(
            max_features=params["max_features"],
            stop_words='english'
        )
    
    vectors = vectorizer.fit_transform(data['tag'])
    
    # Create two types of similarity matrices
    cosine_sim = cosine_similarity(vectors)
    sigmoid_sim = sigmoid_kernel(vectors)
    
    return vectorizer, cosine_sim, sigmoid_sim

def recommend_movies(
    movie_title: str,
    cleaned_df: pd.DataFrame,
    cosine_sim: np.ndarray,
    sigmoid_sim: np.ndarray,
    top_n: int = 5,
    method: str = 'hybrid'
) -> List[Dict[str, Any]]:
    """Generate movie recommendations using multiple similarity measures.
    
    Args:
        movie_title: Title of the movie to base recommendations on
        cleaned_df: DataFrame containing movie data
        cosine_sim: Cosine similarity matrix
        sigmoid_sim: Sigmoid kernel similarity matrix
        top_n: Number of recommendations to return
        method: Recommendation method ('cosine', 'sigmoid', or 'hybrid')
    
    Returns:
        List of dictionaries containing recommended movies and their similarity scores
    """
    try:
        movie_index = cleaned_df[cleaned_df['title'] == movie_title].index[0]
        
        if method == 'cosine':
            distances = cosine_sim[movie_index]
        elif method == 'sigmoid':
            distances = sigmoid_sim[movie_index]
        else:  # hybrid approach
            # Combine both similarity measures with weights
            distances = 0.7 * cosine_sim[movie_index] + 0.3 * sigmoid_sim[movie_index]
        
        # Get top N similar movies
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
        
        recommendations = []
        for i in movies_list:
            recommendations.append({
                'title': cleaned_df.iloc[i[0]].title,
                'similarity': float(i[1]),  # Convert numpy float to Python float for JSON serialization
                'method': method
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
        
        # Try both vectorizer types
        for vectorizer_type in ["count", "tfidf"]:
            # Model parameters
            model_params = {
                "max_features": 5000,
                "vectorizer_type": vectorizer_type,
                "stem_words": True,
                "remove_stopwords": True,
                "ngram_range": (1, 2) if vectorizer_type == "tfidf" else (1, 1)
            }
            mlflow.log_params({f"{vectorizer_type}_{k}": v for k, v in model_params.items()})
            
            # Preprocess data
            cleaned_df = preprocess_data(movies, credits)
            preprocessing_time = time.time() - start_time
            
            # Create model
            model_start_time = time.time()
            vectorizer, cosine_sim, sigmoid_sim = create_recommendation_model(cleaned_df, model_params)
            model_time = time.time() - model_start_time
            
            # Log metrics
            mlflow.log_metrics({
                f"{vectorizer_type}_preprocessing_time_seconds": preprocessing_time,
                f"{vectorizer_type}_model_creation_time_seconds": model_time,
                f"{vectorizer_type}_total_time_seconds": time.time() - start_time,
                f"{vectorizer_type}_feature_count": len(vectorizer.get_feature_names_out()),
                f"{vectorizer_type}_vocabulary_size": len(vectorizer.vocabulary_)
            })
            
            # Save model artifacts
            model_path = f"models/model_{vectorizer_type}_{mlflow.active_run().info.run_id}"
            Path(model_path).mkdir(parents=True, exist_ok=True)
            
            np.save(f"{model_path}/cosine_similarity.npy", cosine_sim)
            np.save(f"{model_path}/sigmoid_similarity.npy", sigmoid_sim)
            pd.to_pickle(vectorizer, f"{model_path}/vectorizer.pkl")
            
            with open(f"{model_path}/model_info.json", "w") as f:
                json.dump({
                    "feature_names": vectorizer.get_feature_names_out().tolist(),
                    "movie_indices": cleaned_df['title'].to_list()
                }, f)
            
            mlflow.log_artifacts(model_path, f"model_{vectorizer_type}")                # Generate example recommendations using different methods
            example_movie = "Avatar"
            method_metrics = {}
            
            for method in ['cosine', 'sigmoid', 'hybrid']:
                recommendations = recommend_movies(
                    example_movie,
                    cleaned_df,
                    cosine_sim,
                    sigmoid_sim,
                    top_n=5,
                    method=method
                )
                
                # Log recommendations and calculate metrics
                if recommendations:
                    recommendations_df = pd.DataFrame(recommendations)
                    
                    # Calculate metrics for this method
                    avg_similarity = recommendations_df['similarity'].mean()
                    max_similarity = recommendations_df['similarity'].max()
                    min_similarity = recommendations_df['similarity'].min()
                    
                    # Log method-specific metrics
                    mlflow.log_metrics({
                        f"{vectorizer_type}_{method}_avg_similarity": avg_similarity,
                        f"{vectorizer_type}_{method}_max_similarity": max_similarity,
                        f"{vectorizer_type}_{method}_min_similarity": min_similarity
                    })
                    
                    # Save recommendations to CSV
                    recommendations_df.to_csv(f"{vectorizer_type}_{method}_recommendations.csv", index=False)
                    mlflow.log_artifact(
                        f"{vectorizer_type}_{method}_recommendations.csv",
                        f"recommendations/{vectorizer_type}"
                    )
                    
                    # Store metrics for comparison
                    method_metrics[method] = {
                        'avg_similarity': avg_similarity,
                        'max_similarity': max_similarity,
                        'min_similarity': min_similarity,
                        'recommendations': recommendations
                    }
                    
                    print(f"\nRecommendations using {vectorizer_type} vectorizer and {method} method:")
                    print(f"Average similarity: {avg_similarity:.4f}")
                    for movie in recommendations:
                        print(f"{movie['title']} (similarity: {movie['similarity']:.4f})")
            
            # Log comparative metrics
            best_method = max(method_metrics.items(), key=lambda x: x[1]['avg_similarity'])[0]
            mlflow.log_param(f"{vectorizer_type}_best_method", best_method)
            
            # Create and log a comparison report
            comparison_data = {
                'method': [],
                'avg_similarity': [],
                'max_similarity': [],
                'min_similarity': []
            }
            
            for method, metrics in method_metrics.items():
                comparison_data['method'].append(method)
                comparison_data['avg_similarity'].append(metrics['avg_similarity'])
                comparison_data['max_similarity'].append(metrics['max_similarity'])
                comparison_data['min_similarity'].append(metrics['min_similarity'])
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(f"{vectorizer_type}_method_comparison.csv", index=False)
            mlflow.log_artifact(
                f"{vectorizer_type}_method_comparison.csv",
                f"comparisons/{vectorizer_type}"
            )

if __name__ == "__main__":
    main()

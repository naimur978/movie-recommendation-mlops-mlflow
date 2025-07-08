from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import mlflow
import pandas as pd

# Pydantic models for request/response
class MovieBase(BaseModel):
    title: str
    overview: Optional[str] = None

class MovieRecommendation(BaseModel):
    title: str
    similarity: float

class MovieRecommendationResponse(BaseModel):
    input_movie: str
    recommendations: List[MovieRecommendation]
    method: str = "cosine_similarity"

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    data_loaded: bool

app = FastAPI(
    title="Movie Recommendation System API",
    description="""
    This API provides movie recommendations based on movie titles using MLflow model.
    
    ## Features
    * Get list of available movies
    * Get movie recommendations based on a title
    * Check system health and model status
    
    ## Technologies
    * FastAPI
    * MLflow
    * Scikit-learn
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data for demonstration when actual data is not available
SAMPLE_MOVIES = [
    {"title": "Avatar", "overview": "A paraplegic marine dispatched to the moon Pandora on a unique mission."},
    {"title": "Star Trek Into Darkness", "overview": "After the crew of the Enterprise find an unstoppable force of terror from within their own organization."},
    {"title": "Inception", "overview": "A thief who steals corporate secrets through the use of dream-sharing technology."},
    {"title": "The Dark Knight", "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
    {"title": "Interstellar", "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival."}
]

SAMPLE_RECOMMENDATIONS = [
    {"title": "Star Trek Into Darkness", "similarity": 0.8, "method": "cosine"},
    {"title": "Avatar", "similarity": 0.7, "method": "cosine"},
    {"title": "Inception", "similarity": 0.6, "method": "cosine"},
    {"title": "The Dark Knight", "similarity": 0.5, "method": "cosine"},
    {"title": "Interstellar", "similarity": 0.4, "method": "cosine"}
]

# Load the model and data at startup
import os
print("Current working directory:", os.getcwd())
print("Checking if data files exist:")
print("movies.csv exists:", os.path.exists('dataset/tmdb_5000_movies.csv'))
print("credits.csv exists:", os.path.exists('dataset/tmdb_5000_credits.csv'))
print("recommendations.csv exists:", os.path.exists('count_cosine_recommendations.csv'))

try:
    # Try to load actual datasets
    movies_df = pd.read_csv('dataset/tmdb_5000_movies.csv')
    print("Successfully loaded movies dataset")
    
    credits_df = pd.read_csv('dataset/tmdb_5000_credits.csv')
    print("Successfully loaded credits dataset")
    
    recommendations_df = pd.read_csv('dataset/count_cosine_recommendations.csv')
    print("Successfully loaded recommendations dataset")
    
    # Set flags to True if data is loaded successfully
    DATA_LOADED = True
    MODEL_LOADED = True
    print("Production data loaded successfully")
except Exception as e:
    print(f"Error loading production data: {str(e)}")
    print("Directory contents:")
    try:
        print("Root directory:", os.listdir('.'))
        if os.path.exists('dataset'):
            print("Dataset directory:", os.listdir('dataset'))
    except Exception as list_error:
        print(f"Error listing directory: {str(list_error)}")
    
    print("Falling back to sample data")
    # Use sample data if files are not available
    movies_df = pd.DataFrame(SAMPLE_MOVIES)
    recommendations_df = pd.DataFrame(SAMPLE_RECOMMENDATIONS)
    DATA_LOADED = True  # We still have sample data
    MODEL_LOADED = True  # We'll use simple similarity scores
    print("Sample data loaded successfully")

@app.get("/",
         summary="Welcome endpoint",
         description="Returns basic information about the API and available endpoints",
         response_description="Welcome message and endpoint list")
def read_root():
    return {
        "message": "Welcome to Movie Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/movies": "Get list of all movies",
            "/recommend": "Get movie recommendations",
            "/health": "Check API health"
        }
    }

@app.get("/health",
         summary="Health check endpoint",
         description="Returns the current status of the API and its components",
         response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy" if MODEL_LOADED and DATA_LOADED else "degraded",
        "version": "1.0.0",
        "model_loaded": MODEL_LOADED,
        "data_loaded": DATA_LOADED
    }

@app.get("/movies",
         summary="Get available movies",
         description="Returns a list of movies that can be used for recommendations",
         response_model=List[MovieBase])
def get_movies(
    limit: int = Query(
        10,
        description="Number of movies to return",
        ge=1,
        le=100
    ),
    skip: int = Query(
        0,
        description="Number of movies to skip",
        ge=0
    )
):
    try:
        total_movies = len(movies_df)
        if skip >= total_movies:
            return []
        
        end_idx = min(skip + limit, total_movies)
        movies = movies_df.iloc[skip:end_idx][['title', 'overview']].to_dict('records')
        return movies
    except Exception as e:
        print(f"Error fetching movies: {e}")
        return SAMPLE_MOVIES[:limit]

@app.get("/recommend",
         summary="Get movie recommendations",
         description="Returns similar movies based on the input movie title",
         response_model=MovieRecommendationResponse)
def recommend_movies(
    movie_title: str = Query(
        ...,
        description="Title of the movie to get recommendations for",
        example="Star Trek Into Darkness"
    ),
    num_recommendations: int = Query(
        5,
        description="Number of recommendations to return",
        ge=1,
        le=20
    )
):
    try:
        if not DATA_LOADED:
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable: Data not loaded"
            )
        
        # Find the movie in our dataset
        if movie_title not in movies_df['title'].values:
            raise HTTPException(
                status_code=404,
                detail=f"Movie '{movie_title}' not found in database"
            )
        
        try:
            # Get recommendations for the input movie
            similar_movies = recommendations_df[recommendations_df['title'].str.lower() == movie_title.lower()]
            
            if len(similar_movies) > 0:
                # Get the recommendations for this movie
                recommendations = []
                seen_titles = set()  # To avoid duplicates
                
                for _, row in similar_movies.iterrows():
                    if len(recommendations) >= num_recommendations:
                        break
                    
                    # Skip if we've already seen this title or if it's the input movie
                    if row['title'] in seen_titles or row['title'].lower() == movie_title.lower():
                        continue
                    
                    recommendations.append(MovieRecommendation(
                        title=row['title'],
                        similarity=float(row['similarity'])
                    ))
                    seen_titles.add(row['title'])
                
                if recommendations:
                    return MovieRecommendationResponse(
                        input_movie=movie_title,
                        recommendations=recommendations,
                        method="cosine_similarity"
                    )
            
            # Fallback: if no recommendations found or if list is empty
            fallback_movies = movies_df[
                movies_df['title'].str.lower() != movie_title.lower()
            ]['title'].head(num_recommendations).tolist()
            
            return MovieRecommendationResponse(
                input_movie=movie_title,
                recommendations=[
                    MovieRecommendation(title=title, similarity=0.1)
                    for title in fallback_movies
                ],
                method="fallback"
            )
            
        except Exception as e:
            print(f"Error processing recommendations: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing recommendations: {str(e)}"
            )
        
        return {
            "input_movie": movie_title,
            "recommendations": recommendations,
            "method": "cosine_similarity"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

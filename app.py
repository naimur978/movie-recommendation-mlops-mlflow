from fastapi import FastAPI, HTTPException, Query
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

# Load the model and data at startup
try:
    # Load your datasets
    movies_df = pd.read_csv('dataset/tmdb_5000_movies.csv')
    credits_df = pd.read_csv('dataset/tmdb_5000_credits.csv')
    recommendations_df = pd.read_csv('count_cosine_recommendations.csv')
    
    # Set flags to True if data is loaded successfully
    DATA_LOADED = True
    MODEL_LOADED = True
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading data: {e}")
    movies_df = pd.DataFrame(columns=['title', 'overview'])
    credits_df = pd.DataFrame()
    recommendations_df = pd.DataFrame(columns=['title', 'similarity', 'method'])
    MODEL_LOADED = False
    DATA_LOADED = False

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
        movies = movies_df.iloc[skip:skip+limit][['title', 'overview']].to_dict('records')
        return movies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

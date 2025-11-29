# movie-recommendation-system
Hybrid recommendation engine
# Movie Recommendation System (Hybrid Model)

## Overview
Hybrid recommender combining content-based (TF-IDF on movie descriptions) and
collaborative filtering (user-rating-based cosine similarity). This simplified demo
includes a small sample dataset and a Flask app to query recommendations.

## Tech Stack
Python, Flask, Pandas, Scikit-Learn

## How to run (local demo)
1. Create and activate virtualenv (recommended):
   python3 -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
2. Install requirements:
   pip install -r requirements.txt
3. Run the demo:
   python app.py
4. Open http://127.0.0.1:5000 and use the example endpoints.

## Files
- data/sample_movies.csv : small dataset used by demo
- src/recommender.py : core hybrid recommender functions
- app.py : Flask demo app
- requirements.txt : Python dependencies

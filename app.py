# app.py - simple Flask demo
from flask import Flask, request, render_template_string, jsonify
import pandas as pd
from src.recommender import HybridRecommender

app = Flask(__name__)
movies = pd.read_csv('data/sample_movies.csv')
recommender = HybridRecommender(movies)

HOME = """<h2>Movie Recommender Demo</h2>
<form action='/recommend' method='get'>
Movie title: <input name='title' value='Toy Story (1995)'><br>
Top N: <input name='n' value='3'><br>
<input type='submit' value='Get Recommendations'>
</form>
"""

@app.route('/')
def home():
    return HOME

@app.route('/recommend')
def recommend():
    title = request.args.get('title','Toy Story (1995)')
    n = int(request.args.get('n',3))
    recs = recommender.recommend_by_title(title, topn=n)
    return jsonify(recs)

if __name__ == '__main__':
    app.run(debug=True)

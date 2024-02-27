from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)

class MovieRecommender:
    def __init__(self):
        self.movie_info = pd.read_csv('final_data.csv')
        self.movie_info['Title'] = self.movie_info['Title'].str.lower()
        # self.movie_info['Title']=str(self.movie_info['Title']).lower()
        print("title------>",self.movie_info['Title'])
        self.tfid = TfidfVectorizer()
        self.vectors = self.tfid.fit_transform(self.movie_info['Genre'])
        self.index = pd.Series(data=self.movie_info.index, index=self.movie_info['Title'])

    def get_movie_names(self, name, n):

        similarity_distance = linear_kernel(self.vectors[self.index[name]], self.vectors)
        similarity_scores = pd.DataFrame(similarity_distance).T
        similarity_scores.columns = ['Scores']
        similarity_scores = similarity_scores.sort_values("Scores", ascending=False)
        movie_list = []
        for i in range(1, n + 1):
            result = {
                'Title': self.movie_info['Title'][similarity_scores.index[i]],
                'Genre': self.movie_info['Genre'][similarity_scores.index[i]]

            }
            movie_list.append(result)
        return movie_list

recommender = MovieRecommender()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movie_name =request.form['movie_name']
        movie_name=movie_name.lower()
        print("movie_name------->",movie_name)
        num_recommendations = int(request.form['num_recommendations'])
        recommendations = recommender.get_movie_names(movie_name, num_recommendations)
        return render_template('index.html', movie_name=movie_name, recommendations=recommendations)
    return render_template('index.html', movie_name='', recommendations=[])

if __name__ == '__main__':
    app.run(debug=True)

import os
import sys
sys.path.append('/home/caleb/RecSys')
sys.path.append('/home/caleb/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import SimilarMovie, Movie

movies = Movie.objects.all().values('id')
movie_ids = set()
for pair in movies:
    movie_ids.add(pair['id'])
for count in range(5, 21):
    total_num = 0
    for movie_id in movie_ids:
        similar_movies = SimilarMovie.objects.filter(movie_id=movie_id).order_by('rank')

        id_set = set()
        algorithm_count = {}
        # Get the list of displayed similar movies for this movie
        for similar in similar_movies:
            # Each algorithm will contribute constant number of movies
            if algorithm_count.get(similar.algorithm, 0) >= count:
                continue
            algorithm_count[similar.algorithm] = algorithm_count.get(similar.algorithm, 0) + 1
            if similar.similar_movie_id not in id_set:
                id_set.add(similar.similar_movie_id)
        total_num += len(id_set)
    print("Count per algo: " + str(count) + " Expectation number: " + str(total_num/len(movie_ids)))
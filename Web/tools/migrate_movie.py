import os
import sys
sys.path.append('/home/caleb/RecSys')
sys.path.append('/home/caleb/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie, NewMovie


for movie in Movie.objects.all():
    newMovie = NewMovie(omdb_id=int(movie.id), imdb_id=int(movie.imdb_id[2:]), movielens_id=int(movie.movielens_id),
                        title=movie.title, year=movie.year, rating=movie.rating, runtime=movie.runtime, released=movie.released, director=movie.director, writer=movie.writer,
                        cast=movie.cast, metacritic=movie.metacritic, imdb_rating=movie.imdb_rating,
                        imdb_votes=movie.imdb_votes, poster=movie.poster, plot=movie.plot, full_plot=movie.full_plot,
                        language=movie.language, country=movie.country, awards=movie.awards,
                        last_updated=movie.last_updated, popularity=movie.popularity, budget=movie.budget,
                        revenue=movie.revenue)
    if movie.tmdb_id:
        newMovie.tmdb_id = int(movie.tmdb_id)
    newMovie.save()
    for genre in movie.genres.all():
        newMovie.genres.add(genre)
    newMovie.save()
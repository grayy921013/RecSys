#! /usr/bin/python3
import os
import sys
sys.path.append('/home/ubuntu/RecSys')
sys.path.append('/home/ubuntu/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie, Genre
f = open("omdbMovies.txt", encoding="ISO-8859-1")
first_line = True
for line in f:
    if first_line:
        first_line = False
        continue
    words = line.split("\t")
    for i in range(0, len(words)):
        words[i] = words[i].strip()
    movies = Movie.objects.filter(id=words[0])
    if len(movies) == 0:
        # No corresponding movie in database
        continue
    movie = movies[0]
    if not words[6]:
        continue
    genres = words[6].split(",")
    for genre_name in genres:
        genre_name = genre_name.strip()
        genre_list = Genre.objects.filter(name=genre_name)
        if len(genre_list) == 0:
            genre = Genre(name=genre_name)
            genre.save()
        else:
            genre = genre_list[0]
        movie.genres.add(genre)
        movie.save()
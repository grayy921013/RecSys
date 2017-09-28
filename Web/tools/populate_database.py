#! /usr/bin/python3
import os
import sys
sys.path.append('/home/ubuntu/RecSys')
sys.path.append('/home/ubuntu/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie
f = open("omdbMovies.txt", encoding="ISO-8859-1")
f2 = open("mlLinks.csv")
first_line = True
imdb_id_set = {}

for line in f2:
    if first_line:
        first_line = False
        continue
    words = line.split(",")
    imdb_id_set["tt" + words[1].strip()] = words[0]

first_line = True
for line in f:
    if first_line:
        first_line = False
        continue
    words = line.split("\t")
    for i in range(0, len(words)):
        words[i] = words[i].strip()
    if not words[1] in imdb_id_set:
        continue
    movie = Movie(id=words[0], imdb_id=words[1], movielens_id=imdb_id_set[words[1]], title=words[2], year=int(words[3]),
                  runtime=words[5], genre=words[6],
                  released=words[7], director=words[8], writer=words[9], cast=words[10],
                  metacritic=words[11], poster=words[14], plot=words[15],
                  full_plot=words[16], language=words[17], country=words[18], awards=words[19],
                  last_updated=words[20])
    try:
        movie.imdb_votes = int(words[13])
    except ValueError:
        print("no votes")
    try:
        movie.rating = float(words[4])
    except ValueError:
        print("no rating")
    try:
        movie.imdb_rating = float(words[12])
    except ValueError:
        print("no rating")

    movie.save()

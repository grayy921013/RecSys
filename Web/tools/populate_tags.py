import os
import sys
sys.path.append('/home/caleb/RecSys')
sys.path.append('/home/caleb/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie

f = open("tags.csv", encoding="ISO-8859-1")
first_line = True
for line in f:
    if first_line:
        first_line = False
        continue
    words = line.split(",")
    movielens_id = int(words[1])
    tag = words[2].strip()
    movies = Movie.objects.filter(movielens_id=movielens_id)
    if not movies:
        continue
    movie = movies[0]
    if movie.tags:
        movie.tags = movie.tags + " " + tag
    else:
        movie.tags = tag
    movie.save()
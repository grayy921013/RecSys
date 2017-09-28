import os
import sys
sys.path.append('/home/ubuntu/RecSys')
sys.path.append('/home/ubuntu/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie


f = open("links.csv", encoding="ISO-8859-1")
first_line = True

for line in f:
    if first_line:
        first_line = False
        continue
    input = line.split(",")
    try:
        movie = Movie.objects.get(imdb_id="tt" + input[1].strip())
        movie.tmdb_id = input[2].strip()
        movie.save()
    except:
        # not in our database
        continue
import sys
import os
import django
import urllib.request
import urllib.error
import os.path
from django.conf import settings

sys.path.append('/home/ubuntu/RecSys')
sys.path.append('/home/ubuntu/RecSys/Web')

os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "Web.settings")
django.setup()

from mainsite.models import Movie

for movie in Movie.objects.all():
    if os.path.exists("../mainsite/static/posts/" + movie.movielens_id + ".jpg"):
        continue
    try:
        urllib.request.urlretrieve(movie.poster, "../mainsite/static/posts_omdb/" + movie.movielens_id + ".jpg")
    except urllib.error.HTTPError:
        print("No poster for " + movie.movielens_id)
    except ValueError:
        print("No poster url for " + movie.movielens_id)
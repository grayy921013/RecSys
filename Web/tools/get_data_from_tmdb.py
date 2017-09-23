import os
import sys
sys.path.append('/home/ubuntu/RecSys')
sys.path.append('/home/ubuntu/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie, Genre
import urllib.request
import urllib.error
import time
import json

for movie in Movie.objects.all():
    while True:
        if not movie.tmdb_id or movie.popularity:
            # no tmdb data
            break
        try:
            with urllib.request.urlopen("https://api.themoviedb.org/3/movie/" + movie.tmdb_id + "?api_key=53d9ee0b6ce3d6cf311edad90ab97e70") as response:
                html = response.read().decode("utf-8")
                data = json.loads(html)
                movie.popularity = float(data["popularity"])
                movie.budget = int(data["budget"])
                movie.revenue = int(data["revenue"])
                for genre in data["genres"]:
                    genreSet = Genre.objects.filter(id=genre["id"])
                    if len(genreSet) == 0:
                        # Create new genre
                        genre = Genre(id=genre["id"], name=genre["name"])
                        genre.save()
                    else:
                        genre = genreSet[0]
                    movie.genres.add(genre)

                movie.save()

                # try to get poster
                if os.path.exists("../mainsite/static/posters/" + movie.movielens_id + ".jpg"):
                    break
                if data["poster_path"]:
                    # try to get new poster
                    poster_url = "https://image.tmdb.org/t/p/w300_and_h450_bestv2" + data["poster_path"]
                    urllib.request.urlretrieve(poster_url, "../mainsite/static/posters/" + movie.movielens_id + ".jpg")
                    print("poster found for " + movie.movielens_id)
        except urllib.error.HTTPError:
            # Sleep one second and retry
            print("Too fast")
            time.sleep(1)
        break


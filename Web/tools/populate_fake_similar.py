import os
import sys
sys.path.append('/home/caleb/RecSys')
sys.path.append('/home/caleb/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import SimilarMovie

for i in range(3, 60):
    similar = SimilarMovie(rank=i, algorithm=-3, movie_id=3, similar_movie_id=i)
    similar.save()
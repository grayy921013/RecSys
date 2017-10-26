import os
import sys
sys.path.append('/home/caleb/RecSys')
sys.path.append('/home/caleb/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import GroundTruth

f = open("groundtruth.exp1-1.csv")
first_line = True
for line in f:
    if first_line:
        first_line = False
        continue
    if line.strip():
        movieid1, movieid2, rating, username = line.strip().split(",")
        try:
            truth = GroundTruth(movielens_id=int(movieid1))
            truth.save()
        except:
            pass
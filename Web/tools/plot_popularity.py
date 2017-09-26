#! /usr/bin/python3
import os
import sys
sys.path.append('/home/ubuntu/RecSys')
sys.path.append('/home/ubuntu/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

movies = Movie.objects.all().order_by('-popularity')
list = []
for movie in movies:
    list.append(movie.popularity)

plt.plot(list)
plt.ylabel('popularity')
plt.savefig('popularity.png')
plt.clf()
plt.plot(list[:2000])
plt.ylabel('popularity')
plt.savefig('popularity-2000.png')
plt.clf()
plt.plot(list[:1000])
plt.ylabel('popularity')
plt.savefig('popularity-1000.png')
plt.clf()
plt.plot(list[:500])
plt.ylabel('popularity')
plt.savefig('popularity-500.png')
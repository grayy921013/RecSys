import os
import sys
sys.path.append('/home/ubuntu/RecSys')
sys.path.append('/home/ubuntu/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie

f2 = open("tomatoes.txt", encoding="ISO-8859-1")
first_line = True

for line in f2:
    if first_line:
        first_line = False
        continue
    input = line.split("\t")
    if not input[12]:
        continue
    try:
        movie = Movie.objects.get(id=input[0])
        boxoffice = input[12].replace("$", "")
        multiplier = 1
        if boxoffice.find("k"):
            multiplier = 1000
            boxoffice = boxoffice.replace("k", "")

        if boxoffice.find("M"):
            multiplier = 1000000
            boxoffice = boxoffice.replace("M", "")

        amount = int(float(boxoffice) * multiplier)
        print(input[0], amount)

        movie.blockbuster = amount
        movie.save()
    except:
        # not in our database
        continue
import os
import sys
sys.path.append('/home/caleb/RecSys')
sys.path.append('/home/caleb/RecSys/Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie
import re
s = "Example String"
replaced = re.sub('[ES]', 'a', s)


for movie in Movie.objects.all():
    writer = movie.writer
    if not writer:
        continue
    replaced = re.sub("\([^,]*\)", "", writer)
    names = replaced.split(",")
    processed_names = []
    for name in names:
        new_name = ""
        words = name.split(" ")
        for word in words:
            if word:
                if not new_name:
                    new_name = word
                else:
                    new_name += "_" + word

        processed_names.append(new_name)
    movie.writer_processed = ", ".join(processed_names)
    movie.save()
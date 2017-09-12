from django.http import HttpResponse, JsonResponse
from mainsite.models import Movie

def get_metadata(request, imdb_id):
    try:
        movie = Movie.objects.get(imdb_id="tt" + imdb_id)
        return JsonResponse(dict(result=movie_to_json(movie)))
    except:
        return HttpResponse("No result found.")

def movie_to_json(movie):
    dict = {}
    dict['title'] = movie.title
    return dict
from django.http import HttpResponse, JsonResponse
from models import Movie, Userinfo, Genre

from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
import math
import random
import random_module


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


def get_random_movie(user_id):
    genre = random_module.random_genre()
    popularity_range = random_module.random_popularity_range()

    movie_list = []
    for i in range(20):
        movie = Movie.objects.filter(genres__name__contains=genre, popularity__gte=popularity_range[1],
                                 popularity__lte=popularity_range[0]).order_by("?").all()[0]
        movie_list.append(movie)

    return movie_list


#### Web UI Implementation####

def userlogin(request):
    print("enter login")
    context = {}

    if request.method == 'GET':
        return render(request, 'login.html', context)

    errors = " "

    if request.method == 'POST' and request.POST['log_username'] and request.POST['log_password']:
        username = request.POST.get('log_username')
        password = request.POST.get('log_password')
        user = authenticate(username=username, password=password)

        if user is not None:

            if user.is_active:
                print("user valid")
                login(request, user)
                return HttpResponseRedirect('/')
            else:
                print("user disabled")
                errors = 'User disabled. Please try another.'

        else:
            print("Invalid input")
            errors = "User does not exist. Please try again."

    else:
        errors = 'Something went wrong...'

    context['errors'] = errors
    return render(request, 'login.html', context)


def register(request):
    context = {}

    if request.method == 'GET':
        return render(request, 'register.html', context)

    errors = []
    context['errors'] = errors

    registered = False
    if not 'username' in request.POST or not request.POST['username']:
        errors.append('Invalid username.')

    if not 'password' in request.POST or not request.POST['password']:
        errors.append('Invalid password.')

    if not 'confirm_password' in request.POST or not request.POST['confirm_password']:
        errors.append('Invalid password confirm.')

    else:
        username = request.POST.get('username')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # other info
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        education = request.POST.get('education')
        employment = request.POST.get('employment')

        # check password
        if password != confirm_password:
            errors.append('The passwords did not match. Please check.')

        elif not errors:
            user = User.objects.create_user(username=username, password=password)
            user.save()
            new_info = Userinfo(user=user, age=age, gender=gender, education=education, employment=employment)
            new_info.save()

            user = authenticate(username=username, password=password)
            login(request, user)

            return HttpResponseRedirect('/')

    print(errors)
    return render(request, 'register.html', {'errors': errors})


@login_required
def home(request):
    # get the random movie
    random_movie_list = get_random_movie(request.user.id)

    errors = " "
    if request.method == "GET":
        context = {
            'errors': errors,
            'movie_list': random_movie_list,
            'titile': 'Home Page'
        }
        return render(request, 'home.html', context)


@login_required
def blockbuster(request):
    # get the random movie
    random_movie_list = Movie.objects.exclude(popularity__isnull=True).order_by('-popularity')[:30]

    errors = " "
    if request.method == "GET":
        context = {
            'errors': errors,
            'movie_list': random_movie_list,
            'title': 'Block Buster'
        }
        return render(request, 'home.html', context)


@login_required
def profile(request, id):
    try:
        current_user = User.objects.get(id=id)
        context = {'current_user': current_user}
        return render(request, "profile.html", context)
    except ObjectDoesNotExist:
        return render(request, "home.html", context)


@login_required
def userlogout(request):
    logout(request)
    return render(request, "login.html")

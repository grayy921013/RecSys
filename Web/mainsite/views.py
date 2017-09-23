from django.http import HttpResponse, JsonResponse
from mainsite.models import Movie, Userinfo

from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
import math

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
	rand_list = Movie.objects.order_by('?').all()[:20]
	return rand_list


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

        print("username   ", username)
        print("age   ", age)
        print("gender   ", gender)
        print("education   ", education)
        print("employment   ", employment)

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
    # prepare the movie into nested list, which correspond to the display grid
    row_length = 4
    grid_list = []
    for row in range(0, int(math.ceil(float(len(random_movie_list)) / row_length))):
        row_list = []
        for col in range(0, min(row_length, len(random_movie_list) - (row * row_length))):
            row_list.append(random_movie_list[row * row_length + col])
        grid_list.append(row_list) 
    print(grid_list)

    errors = " "
    if request.method == "GET":
        context = {
            'errors': errors,
            'movie_grid': grid_list 
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

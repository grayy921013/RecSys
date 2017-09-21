from django.http import HttpResponse, JsonResponse
from mainsite.models import Movie

from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User

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

#### Web UI Implementation####

def userlogin(request):
    print("enter login")
    context = {}

    if request.method == 'GET':
        return render(request, 'login.html', context)

    errors = " "

    if request.method == 'POST' and request.POST['log_username'] and request.POST['log_password']:
        print("enter login")
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

            user = authenticate(username=username, password=password)
            login(request, user)

            return HttpResponseRedirect('/')

    print(errors)
    return render(request, 'register.html', {'errors': errors})


@login_required
def home(request):
    print("enter globalStream")
    errors = " "
    if request.method == "GET":
        context = {'errors': errors}
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

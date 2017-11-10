from django.http import HttpResponse, JsonResponse
from mainsite.models import Movie, Userinfo, Genre, PasswordReset, SimilarMovie, UserVote, SearchAction, VotedMovie,\
    SearchStopword, GroundTruth

from django.core.exceptions import ObjectDoesNotExist
from django.shortcuts import render, HttpResponseRedirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.urlresolvers import reverse
import math
import random
import time
from mainsite.random_module import *
import hashlib
import string
from django.http import Http404
from django.utils import timezone
import json

def get_metadata(request, imdb_id):
    try:
        movie = Movie.objects.get(imdb_id)
        return JsonResponse(dict(result=movie_to_json(movie)))
    except:
        return HttpResponse("No result found.")


def movie_to_json(movie):
    dict = {}
    dict['title'] = movie.title
    return dict


def get_random_movie(user_id):

    movie_list = []
    movie_id_set = set()
    groundtruth_movielens_ids = GroundTruth.objects.all().values("movielens_id")
    while True:
        from_groundtruth = random_groundtruth()
        if from_groundtruth:
            print("from ground truth!")
            movie_qs = Movie.objects.filter(movielens_id__in=groundtruth_movielens_ids).exclude(id__in=movie_id_set)[:1]
        else:
            genre = random_genre()
            popularity_range = random_popularity_range()
            movie_qs = Movie.objects.filter(genres__name__contains=genre, popularity__gte=popularity_range[1],
                                     popularity__lte=popularity_range[0]).order_by("?").exclude(id__in=movie_id_set)[:1]
        if not movie_qs.exists():
            continue
        movie = movie_qs[0]
        # Filter out movies that has been voted by more than four users.
        if UserVote.objects.filter(movie1=movie).distinct('user').count() > 4:
            continue
        if movie.id in movie_id_set:
            continue
        movie_id_set.add(movie.id)
        movie_list.append(movie)
        if len(movie_list) == 20:
            break

    movie_id_list = []
    for movie in movie_list:
        movie_id_list.append(movie.id)
    return movie_list, movie_id_list


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

    if not 'username' in request.POST or not request.POST['username']:
        errors.append('Invalid username.')

    if not 'password' in request.POST or not request.POST['password']:
        errors.append('Invalid password.')

    if not 'confirm_password' in request.POST or not request.POST['confirm_password']:
        errors.append('Invalid password confirm.')

    if not 'security-question' in request.POST or not request.POST['security-question']:
        errors.append('Security question is required.')

    if not 'security-answer' in request.POST or not request.POST['security-answer']:
        errors.append('Security answer is required.')

    if not 'consent' in request.POST or not request.POST['consent']:
        errors.append('Please check the privacy statement.')

    username = request.POST.get('username')
    password = request.POST.get('password')
    confirm_password = request.POST.get('confirm_password')

    # other info
    age = request.POST.get('age')
    gender = request.POST.get('gender')
    education = request.POST.get('education')
    employment = request.POST.get('employment')
    security_question = request.POST.get('security-question')
    security_answer = request.POST.get('security-answer')

    # check password
    if password != confirm_password:
        errors.append('The passwords did not match. Please check.')

    if User.objects.filter(username__exact=username):
        errors.append('Username is already taken.')

    if not errors:
        user = User.objects.create_user(username=username, password=password)
        user.save()
        digest = hashlib.sha256(security_answer.encode('utf-8')).hexdigest()
        new_info = Userinfo(user=user, gender=gender, education=education, employment=employment,
                            security_question=security_question, security_answer=digest)
        if age:
            new_info.age = age
        new_info.save()

        user = authenticate(username=username, password=password)
        login(request, user)

        return HttpResponseRedirect(reverse('home'))

    print(errors)
    return render(request, 'register.html', {'errors': errors})


def reset_pwd(request):
    context = {}

    if request.method == 'GET':
        return render(request, 'reset_pwd.html', context)

    errors = []
    context['errors'] = errors

    if not 'username' in request.POST or not request.POST['username']:
        errors.append('Invalid username.')

    if not 'security-question' in request.POST or not request.POST['security-question']:
        errors.append('Security question is required.')

    if not 'security-answer' in request.POST or not request.POST['security-answer']:
        errors.append('Security answer is required.')

    username = request.POST.get('username')
    security_question = request.POST.get('security-question')
    security_answer = request.POST.get('security-answer')

    if not errors:
        user = User.objects.filter(username=username)
        if not user or user[0].userinfo.security_question != security_question or \
                        hashlib.sha256(security_answer.encode('utf-8')).hexdigest() != user[0].userinfo.security_answer:
            errors.append('Invalid username or answer')
            return render(request, 'reset_pwd.html', {'errors': errors})
        token_set = PasswordReset.objects.filter(user=user[0])
        if token_set:
            token = token_set[0]
        else:
            token = PasswordReset(user=user[0])
        token.token = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(20))
        token.save()
        return HttpResponseRedirect(reverse('modifypwd', args=[user[0].id, token.token]))

    print(errors)
    return render(request, 'register.html', {'errors': errors})

def modify_pwd(request, uid, token):
    context = {}

    user_set = User.objects.filter(id=uid)
    if not user_set:
        raise Http404
    user = user_set[0]
    token_set = user.passwordreset_set.all()
    if not token_set:
        raise Http404
    token_db = token_set[0]
    if token_db.token != token or (timezone.now() - token_db.updated_at).total_seconds() > 300:
        raise Http404

    if request.method == 'GET':
        return render(request, 'modify_pwd.html', context)

    errors = []
    context['errors'] = errors


    if not 'password' in request.POST or not request.POST['password']:
        errors.append('Invalid password.')

    if not 'confirm_password' in request.POST or not request.POST['confirm_password']:
        errors.append('Invalid password confirm.')

    password = request.POST.get('password')
    confirm_password = request.POST.get('confirm_password')

    if password != confirm_password:
        errors.append('The passwords did not match. Please check.')

    if not errors:
        user.set_password(password)
        user.save()
        PasswordReset.objects.filter(user_id=uid).delete()
        return HttpResponseRedirect(reverse('home'))

    print(errors)
    return render(request, 'modify_pwd.html', {'errors': errors})


@login_required
def home(request):
    if request.method != "GET":
        raise Http404
    # get the random movie
    if 'movie_list' in request.COOKIES:
        movie_id_list = json.loads(request.COOKIES['movie_list'])
        random_movie_list = list(Movie.objects.filter(id__in=movie_id_list))
        random_movie_list.sort(key=lambda t: movie_id_list.index(t.id))
    else:
        random_movie_list, movie_id_list = get_random_movie(request.user.id)

    finished_movies = VotedMovie.objects.filter(user=request.user, movie__in=random_movie_list, finished=True)
    ongoing_movies = VotedMovie.objects.filter(user=request.user, movie__in=random_movie_list, finished=False)
    finished_ids = set()
    ongoing_ids = set()
    for v in finished_movies:
        finished_ids.add(v.movie_id)
    for v in ongoing_movies:
        ongoing_ids.add(v.movie_id)
    errors = " "
    context = {
        'errors': errors,
        'movie_list': random_movie_list,
        'title': 'Home Page',
        'finished_movies': finished_ids,
        'ongoing_movies': ongoing_ids
    }
    response = render(request, 'home.html', context)
    response.set_cookie('movie_list', json.dumps(movie_id_list))
    return response


@login_required
def refresh(request):
    response = HttpResponseRedirect(reverse('home'))
    response.delete_cookie('movie_list')
    return response


@login_required
def blockbuster(request):
    # get the random movie
    random_movie_list = Movie.objects.exclude(popularity__isnull=True).order_by('-popularity')[:30]
    tmp = []
    for movie in random_movie_list:
        tmp.append(movie)
    hash_obj = hashlib.sha256()
    hash_obj.update(str(request.user.pk).encode('utf-8'))

    seed = hash_obj.hexdigest()

    random.seed(seed)

    random.shuffle(tmp)
    random_movie_list = tmp
    finished_movies = VotedMovie.objects.filter(user=request.user, movie__in=random_movie_list, finished=True)
    ongoing_movies = VotedMovie.objects.filter(user=request.user, movie__in=random_movie_list, finished=False)
    finished_ids = set()
    ongoing_ids = set()
    for v in finished_movies:
        finished_ids.add(v.movie_id)
    for v in ongoing_movies:
        ongoing_ids.add(v.movie_id)

    errors = " "
    if request.method == "GET":
        context = {
            'errors': errors,
            'movie_list': random_movie_list,
            'title': 'Block Buster',
            'finished_movies': finished_ids,
            'ongoing_movies': ongoing_ids
        }
        return render(request, 'home.html', context)


@login_required
def label(request, id):
    errors = " "
    try:
        # get the random movie
        temp_similar_list = Movie.objects.order_by('?')[:30]
        movie_obj = Movie.objects.get(id=id)

        context = {
            'movie': movie_obj,
            'similar_list': temp_similar_list,
            'errors': errors
        }
        return render(request, "label.html", context)
    except ObjectDoesNotExist:
        context = {
            'errors': errors
        }
        return render(request, "home.html", context)


@login_required
def profile(request):
    try:
        current_user = request.user
        num_labels = UserVote.objects.filter(user=current_user).count()
        num_movies = UserVote.objects.filter(user=current_user).values('movie1').distinct().count()

        # for badging layout
        top_percent = 5
        num_level = 1 + (num_movies//10)
        dist_next_level = (num_level * 10) - num_movies
        if num_level == 1:
            badge_level = "glyphicon-pawn"
        elif num_level == 2:
            badge_level = "glyphicon-knight"
        elif num_level == 3:
            badge_level = "glyphicon-bishop"
        elif num_level == 3:
            badge_level = "glyphicon-queen"
        else:
            badge_level = "glyphicon-king"

        voting_list = VotedMovie.objects.filter(user=current_user, finished=False)
        voting_movies = Movie.objects.filter(pk__in=voting_list.values('movie_id'))
        finished_list = VotedMovie.objects.filter(user=current_user, finished=True)
        finished_movies = Movie.objects.filter(pk__in=finished_list.values('movie_id'))
        context = {
            'current_user': current_user,
            'num_labels': num_labels,
            'num_movies': num_movies,
            'voting_movies': voting_movies,
            'finished_movies': finished_movies,
            'top_percent': top_percent,
            'num_level': num_level,
            'badge_level': badge_level,
            'dist_next_level': dist_next_level,
        }
        return render(request, "profile.html", context)
    except ObjectDoesNotExist:
        return render(request, "home.html", {})

@login_required
def search(request):
    if not 'keyword' in request.POST or not request.POST['keyword']:
        return render(request, "home.html", {})
    keyword = request.POST['keyword']
    keyword_list = keyword.strip().split(" ")
    query_set = Movie.objects
    stopword_set = set()
    stopwords = SearchStopword.objects.all()
    for stopword in stopwords:
        stopword_set.add(stopword.word)
    filtered_set = set()
    query_valid = False
    for word in keyword_list:
        if word not in stopword_set:
            query_set = query_set.filter(title__icontains=word)
            query_valid = True
        else:
            filtered_set.add(word)
    if query_valid:
        movie_list = query_set.order_by('-popularity')[:30]
    else:
        movie_list = []
    notice = ""
    if filtered_set:
        for filtered in filtered_set:
            notice += '"' + filtered + '"' + ","
            notice = notice[:-1] + (" is filtered." if len(filtered_set) == 1 else "are filtered.")
    context = {
        'movie_list': movie_list,
        'title': 'Search',
        'empty_msg': "There are no movies found according to your query",
        'notice': notice
    }
    SearchAction.objects.create(user=request.user, keyword=keyword)
    return render(request, 'home.html', context)


@login_required
def get_similar_movies(request, id):
    movies = Movie.objects.filter(id=id)
    movie_list = []
    if not movies:
        return JsonResponse(dict(data=movie_list))
    movie = movies[0]
    similar_movies = SimilarMovie.objects.filter(movie=movie)
    vote_status = {}
    id_set = set()
    algorithm_count = {}
    voted_list = UserVote.objects.filter(user=request.user, movie1_id=id)
    for voted in voted_list:
        vote_status[voted.movie2_id] = voted.action
    for similar in similar_movies:
        if similar.similar_movie_id not in id_set:
            if algorithm_count.get(similar.algorithm, 0) >= 5:
                continue
            algorithm_count[similar.algorithm] = algorithm_count.get(similar.algorithm, 0) + 1
            id_set.add(similar.similar_movie_id)
            similar_movie = similar.similar_movie
            status = 2
            if similar.similar_movie_id in vote_status:
                status = vote_status[similar.similar_movie_id]
            # status: -1 for not similar, 0 for skip, 1 for similar, 2 for not yet voted
            record = {"id": similar_movie.id, "poster": "/static/posters/" + str(similar_movie.movielens_id) + ".jpg",
                      "plot": similar_movie.plot, "title": similar_movie.title, "year": similar_movie.year,
                      "status": status}
            movie_list.append(record)

    # generating seed using movie_id and user_id
    movie_id = id
    user_id = request.user.id

    hash_obj = hashlib.sha256()
    hash_obj.update(('%s%s' % (str(movie_id), user_id)).encode('utf-8'))

    seed = hash_obj.hexdigest()

    random.seed(seed)

    random.shuffle(movie_list)

    return JsonResponse(dict(data=movie_list))


@login_required
def user_vote(request):
    if request.method == "GET":
        raise Http404
    if not request.POST["movie1_id"] or not request.POST["movie2_id"] or not request.POST["action"]:
        return JsonResponse(dict(error="wrong format"))
    vote_set = UserVote.objects.filter(user=request.user, movie1_id=int(request.POST["movie1_id"]),
                                       movie2_id=int(request.POST["movie2_id"]))
    if vote_set:
        vote = vote_set[0]
    else:
        vote = UserVote(user=request.user, movie1_id=int(request.POST["movie1_id"]),
                        movie2_id=int(request.POST["movie2_id"]))
    vote.action = int(request.POST["action"])
    vote.save()
    if vote.action == 2:
        vote.delete()

    movie_id = int(request.POST["movie1_id"])
    similar_movies = SimilarMovie.objects.filter(movie_id=movie_id)
    voted_list = UserVote.objects.filter(user=request.user, movie1_id=movie_id)
    voted_ids = set()
    all_voted = True
    for voted in voted_list:
        voted_ids.add(voted.movie2_id)

    id_set = set()
    algorithm_count = {}
    for similar in similar_movies:
        if similar.similar_movie_id not in id_set:
            if algorithm_count.get(similar.algorithm, 0) >= 5:
                continue
            algorithm_count[similar.algorithm] = algorithm_count.get(similar.algorithm, 0) + 1
            id_set.add(similar.similar_movie_id)

    for id in id_set:
        if id not in voted_ids:
            all_voted = False
            break

    voted_movie_set = VotedMovie.objects.filter(user=request.user, movie_id=movie_id)
    if not voted_movie_set:
        voted_movie = VotedMovie(user=request.user, movie_id=movie_id)
    else:
        voted_movie = voted_movie_set[0]
    voted_movie.finished = all_voted
    voted_movie.save()
    return JsonResponse(dict(success=True))


@login_required
def userlogout(request):
    logout(request)
    return render(request, "login.html")


def check_visited(request):
    path = request.GET['path']
    user_info = request.user.userinfo
    if path == 'index':
        if user_info.has_visited_home:
            return HttpResponse('true')
        else:
            return HttpResponse('false')
    else:
        if user_info.has_visited_label:
            return HttpResponse('true')
        else:
            return HttpResponse('false')


def visit_page(request):
    if 'path' not in request.GET:
        return HttpResponse('invalid path')
    path = request.GET['path']
    user_info = request.user.userinfo
    if path == 'index':
        user_info.has_visited_home = True
        user_info.save()
    else:
        user_info.has_visited_label = True
        user_info.save()
    return HttpResponse('ok')


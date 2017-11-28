# RecSys
Frontend Setup:

1. Clone the repo
2. Make sure you have python3 and pip installed. Install required packages in `Web/requirements.txt` using `pip install -r requirements.txt`. You can also use `virtualenv`.
3. Run `python3 manage.py makemigrations` and `python3 manage.py migrate`.
4. Import all the necessary data in `import.sql`. The sql file is avialable in Google drive, because it is too large to fit in Git.
5. To get the movie posters ready, you need to download poster images from the original server and put them under `Web/static/posters`.
5. Run `python3 manage.py runserver 0.0.0.0:8000` to start a local server.
6. To run locally, you may need to comment the last three lines in `Web/settings.py` because they force the whole site to run with HTTPS.
7. To make the site run with HTTPS, you need to first [configure Apache to run with Django](https://docs.djangoproject.com/en/1.11/howto/deployment/wsgi/modwsgi/). You also need to configure the certificate.

import urllib.request

for filename in open("posters.txt"):
    filename = filename.strip()
    urllib.request.urlretrieve("http://img.loud.ninja/Posters/" + filename, "../mainsite/static/posts/" + filename)

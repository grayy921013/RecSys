import random


'''
Short              2
Documentary   9
Comedy        31
Adventure     9
Fantasy
Action        13
Crime         11
Drama         49
Horror        10
Animation          2
History            2
Mystery            2
Romance       15
War
Sport
Biography
Sci-Fi        7
Western
Family
Thriller      15
Music
Musical
Film-Noir
Adult
Talk-Show
News
Reality-TV

percentage for each genre
Drama: 49
Comedy: 31
Thriller: 15
Romance: 15
Action: 13
Crime: 11
Horror: 10
Documentary: 9
Adventure: 9
Sci-Fi: 7
'''

genre_name = ["Drama", "Comedy", "Thriller", "Action", "Crime", "Horror", "Documentary", "Adventure", "Sci-Fi"]

genre_percentage = [49, 31, 15, 13, 11, 10, 9, 9, 9]


def random_genre():
    total = 0
    for percent in genre_percentage:
        total += percent

    random_num = random.uniform(0, total)

    current_sum = 0
    for i in range(len(genre_percentage)):
        current_sum += genre_percentage[i]
        if current_sum >= random_num:
            return genre_name[i]

    return genre_name[-1]


'''
probability for each popular score range:
score >= 50: 1/3
50 > score >= 5: 1/3
5 > score >= 0: 1/3
'''

popularity_ranges = [[10000, 50], [50, 5], [5, 0]]
popularity_percentage = [1, 1, 1]


def random_popularity_range():
    total = 0
    for percent in popularity_percentage:
        total += percent

    random_num = random.uniform(0, total)
    current_sum = 0
    for i in range(len(popularity_percentage)):
        current_sum += popularity_percentage[i]
        if current_sum >= random_num:
            return popularity_ranges[i]
    return popularity_ranges[-1]


if __name__ == '__main__':
    print random_genre()
    print random_popularity_range()

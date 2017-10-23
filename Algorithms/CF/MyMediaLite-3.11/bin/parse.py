
f = open('model.txt')
f_movies = open('model_movies.txt','w')

i = 0
i2 = 0
line = '123'
while len(line) > 1:
    line = f.readline()
    i += 1

print i
movies, features = list(map(int,f.readline().split()))
for i in range(movies):
    m_features = [None]*features
    m_id = 0
    counter = 0
    for idx in range(features):
        m_id, m_feature, m_value = f.readline().split()
        m_features[idx] = m_value
        counter += float(m_value)
    if counter:
        m_features = [m_id] + m_features 
        f_movies.write(','.join(m_features) + '\n')
f.close()
f_movies.close()
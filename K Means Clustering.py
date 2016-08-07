from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans

fil = open('plants.data')
data = []
for row in fil.read().split('\n'):
    values=row.split(',')
    cur = {}
    for state in values[1:]:
        cur[state]=True
    data.append(cur)

model = KMeans(init='k-means++', n_clusters=10, n_init=10, max_iter=250)
transformed=DictVectorizer(sparse = False).fit_transform(data)
model.fit(transformed)
print model.cluster_centers_

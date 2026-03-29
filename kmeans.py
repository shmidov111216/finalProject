import sys


def parse_vector_list(data_str):
    rows = data_str.split('\n')
    vectors = [row.split(',') for row in rows if row != '']
    dim = len(vectors[0])

    for i in range(len(vectors)):
        for j in range(dim):
            vectors[i][j] = float(vectors[i][j])

    vectors = [tuple(vector) for vector in vectors]
    return vectors


def distance(vec1, vec2):
    return sum([(num1 - num2) ** 2 for num1, num2 in zip(vec1, vec2)]) ** 0.5


def get_kmeans(vectors, k, iter_count=400, epsilon=0.001, formatted=True):
    centroids = vectors[:k]
    is_converged_list = [False for _ in range(k)]
    is_all_converged = False

    i = 0

    while not is_all_converged and i < iter_count:
        if i != 0:
            is_all_converged = all(is_converged_list)

        clusters = [[] for _ in range(k)]

        for vector in vectors:
            closest_centroid = 0
            min_distance = distance(vector, centroids[0])

            for j in range(k):
                cur_distance = distance(vector, centroids[j])
                if cur_distance < min_distance:
                    min_distance = cur_distance
                    closest_centroid = j

            clusters[closest_centroid].append(vector)

        for j in range(k):
            new_centroid = update_centroid(clusters[j])
            if distance(new_centroid, centroids[j]) < epsilon:
                is_converged_list[j] = True
            centroids[j] = new_centroid

        i += 1
    if formatted:
        return '\n'.join([','.join([f"{x:.4f}" for x in tup]) for tup in centroids])
    else:
        return centroids

# sum two vectors, returns new vector
def sum_points(point, other):
    new_point = list(point)
    for i in range(len(point)):
        new_point[i] += other[i]
    return tuple(new_point)


def update_centroid(cluster):
    cluster_sum = list(cluster[0])

    for i in range(1, len(cluster)):
        cluster_sum = list(sum_points(cluster_sum, cluster[i]))

    for i in range(len(cluster_sum)):
        cluster_sum[i] /= len(cluster)

    return tuple(cluster_sum)

def assign_clusters(vectors, centroids):
    labels = []

    for vector in vectors:
        closest_centroid = 0
        min_distance = distance(vector, centroids[0])

        for j in range(1, len(centroids)):
            cur_distance = distance(vector, centroids[j])
            if cur_distance < min_distance:
                min_distance = cur_distance
                closest_centroid = j

        labels.append(closest_centroid)

    return labels

def main(*args):
    if len(args) < 1 or len(args) > 2:
        raise Exception("An Error Has Occurred")
    
    try:
        k = int(args[0])
    except ValueError:
        raise Exception("Incorrect number of clusters!")
    except:
        raise Exception("An Error Has Occurred")

    # default value
    try:
        iter_count = int(args[1]) if len(args)>1 else 400
    except:
        raise Exception("Incorrect maximum iteration!")
    
    vectors = parse_vector_list(sys.stdin.read())
    N = len(vectors)

    if not (1 < k < N):
        raise Exception("Incorrect number of clusters!")
    if iter_count >= 800 or iter_count <= 1:
        raise Exception("Incorrect maximum iteration!")

    try:
        means = get_kmeans(vectors, k, iter_count)
        print(means)
    except:
        raise Exception("An Error Has Occurred")

# python3 main.py 3 50 < input.txt
if __name__ == '__main__':
    main(*sys.argv[1:])

import numpy as np
from math import sqrt
from numba import autojit, double, int_

def gen_data_table(n_users, n_items):
    p0 = 0.4
    pn = (1 - p0)/5
    data_table = np.random.choice([0.0, 1, 2, 3, 4, 5],
                                  n_users * n_items,
                                  p=[p0, pn, pn, pn, pn, pn])
    return data_table.reshape(n_users, n_items)


def get_item_vector(data, item):
    return data[:, item]


def get_user_vector(data, user):
    return data[user]


def get_vector_mean(vector):
    return vector[vector > 0].mean()


def get_intersection_of_vectors(vector_a, vector_b):
    joints = 0
    for i in range(min(len(vector_a), len(vector_b))):
        if vector_a[i] != 0 and vector_b[i] != 0:
            joints += 1
    return joints

@autojit
def sim_cos(vector_a, vector_b):
    dot_prod = 0.0
    magn_a = 0.0
    magn_b = 0.0
    mean_a = get_vector_mean(vector_a)
    mean_b = get_vector_mean(vector_b)
    min_vector_len = min(len(vector_a), len(vector_b))
    joint_factor = get_intersection_of_vectors(vector_a, vector_b) / min_vector_len
    for i in range(min_vector_len):
        if vector_a[i] != 0 and vector_b[i] != 0:
            a = (vector_a[i] - mean_a) * joint_factor
            b = (vector_b[i] - mean_b) * joint_factor
            magn_a += a ** 2
            magn_b += b ** 2
            dot_prod += a * b
    #print(dot_prod, magn_a, magn_b, mean_a, mean_b)
    denominator = sqrt(magn_a * magn_b)
    if denominator == 0:
        return 0.0
    else:
        return dot_prod / denominator


def item_sim(data, item_a, item_b):
    return sim_cos(data[:, item_a], data[:, item_b])


def user_sim(data, user_a, user_b):
    return sim_cos(data[user_a], data[user_b])


def most_similar_users(data, person, number_of_users=5):
    if number_of_users >= len(data):
        number_of_users = len(data) - 2
    scores = [(user_sim(data, person, other_person), other_person)
              for other_person in range(len(data)) if other_person != person
             ]
    scores.sort(reverse=True)
    return scores[0:number_of_users]


def most_similar_items(data, item, number_of_items=5):
    scores = [(item_sim(data, item, other_item), other_item)
              for other_item in range(len(data[0])) if other_item != item
             ]
    scores.sort(reverse=True)
    return scores[0:number_of_items]

def base_line_estimation(data, user, item, avg=float("nan")):
   if avg == float("nan"):
       avg = np.sum(data)/np.count_nonzero(data)
   user_vector = get_user_vector(data, user)
   item_vector = get_item_vector(data, item)
   avg_user = np.sum(user_vector)/np.count_nonzero(user_vector)
   avg_item = np.sum(item_vector)/np.count_nonzero(item_vector)
   return avg_user + avg_item - avg


def recommendations_by_items(data, person, predict_items=[], max_similar_items=7, allow_base_line=False):
    user_vector = get_user_vector(data, person)
    recommendations = []
    if len(predict_items) == 0:
        predict_items = np.argwhere(user_vector == 0).ravel()
    rated_items = np.argwhere(user_vector != 0).ravel()
    total_rating_avg = np.sum(data)/np.count_nonzero(data)
    for i in predict_items:
        sum_sim = 0.0
        sum_sim_weight = 0.0
        item_base_line = 0.0
        if allow_base_line:
            item_base_line = base_line_estimation(data, person, item=i, avg=total_rating_avg)
        most_rated_items = sorted(rated_items, key=lambda x: item_sim(data, x, i), reverse=True)
        if len(most_rated_items) > max_similar_items:
            most_rated_items = most_rated_items[0:max_similar_items]
        # print(i, rated_items)
        for j in most_rated_items:
            if i != j:
                sim = item_sim(data, i, j)
                if sim > 0:
                    sum_sim += sim
                    neighbour_base_line = 0.0
                    if allow_base_line:
                        neighbour_base_line = base_line_estimation(data, person, item=j, avg=total_rating_avg)
                    sum_sim_weight += sim * (user_vector[j] - neighbour_base_line)
        rating = 0.0
        if sum_sim > 0:
            rating = item_base_line + sum_sim_weight / sum_sim
        recommendations.append((rating, i))
    recommendations.sort(reverse=True)
    return recommendations

def recommendations_by_users(data, person, max_similar_users=5):
    user_vector = get_user_vector(data, person)
    recommendations = []
    zero_items = np.argwhere(user_vector == 0).ravel()
    for i in zero_items:
        sum_sim = 0.0
        sum_sim_weight = 0.0
        scores = [(user_sim(data, person, other_person), other_person)
                  for other_person in range(len(data))
                      if other_person != person and data[other_person, i] != 0
                 ]
        scores.sort(reverse=True)
        if len(scores) > max_similar_users:
            scores = scores[0:max_similar_users]
        for user_score, user_idx in scores:
            sim = user_score
            if sim > 0:
                sum_sim += sim
                sum_sim_weight += sim * data[user_idx, i]
        rating = 0.0
        if sum_sim > 0:
            rating = sum_sim_weight / sum_sim
        recommendations.append((rating, i))
    recommendations.sort(reverse=True)
    return recommendations

def get_RMSE(data, user):
    user_vector = get_user_vector(data, user)
    rated_items = np.argwhere(user_vector != 0).ravel()
    recs = recommendations_by_items(data, user, rated_items, max_similar_items=10)
    rmse = 0.0
    if len(recs) > 0:
        for r, i in recs:
            rmse += (user_vector[i] - r) ** 2
            # print(user_vector[i], r, i)
        rmse = sqrt(rmse / len(recs))
    return rmse, recs

def matrix_rmse(src_matrix, clc_matrix):
    if src_matrix.shape != clc_matrix.shape or src_matrix.size == 0:
        return float("inf")
    rmse = 0.0
    for i in range(src_matrix.size):
        if src_matrix.flat[i] > 0:
            rmse += (src_matrix.flat[i] - clc_matrix.flat[i]) ** 2
    return sqrt(rmse / src_matrix.size)


# https://gist.github.com/jhemann/5584536
@autojit(locals={'step': int_, 'e': double, 'alpha': double})
def matrix_factorization(R, P, Q, K):
    steps = 5000
    alpha = 0.0001
    beta = 0.02
    half_beta = beta / 2.0
    N, M = R.shape
    e = 0.0
    for step in range(steps):
        for i in range(N):
            for j in range(M):
                if R[i,j] > 0:
                    eij = R[i,j]
                    for p in range(K):
                        eij -= P[i,p] * Q[j,p]
                    for k in range(K):
                        pik = P[i,k]
                        qjk = Q[j,k]
                        P[i,k] += alpha * (2 * eij * qjk - beta * pik)
                        Q[j,k] += alpha * (2 * eij * pik - beta * qjk)
        e = 0.0
        for i in range(N):
            for j in range(M):
                if R[i,j] > 0:
                    rij = R[i,j]
                    for p in range(K):
                        rij -= P[i,p] * Q[j,p]
                    e = e + rij * rij
                    for k in range(K):
                        e += half_beta * (P[i,k]*P[i,k] + Q[j,k]*Q[j,k])
        if e < 0.001:
            break
    return P, Q, step + 1, e

def test_recommender():
    data = gen_data_table(n_users=20, n_items=40)
    np.set_printoptions(linewidth=170)
    print(data)

    # print(get_item_vector(rating_data, 1))
    # print(get_item_vector(rating_data, 3))
    # print(item_sim(rating_data, 1, 3))

    curr_user = 0
    for similar_user in most_similar_users(data, curr_user, 5):
        print("user [%d] (%f)" % (similar_user[1], similar_user[0]))
        curr_user_vector = get_user_vector(data, curr_user)
        similar_user_vector = get_user_vector(data, similar_user[1])
        print("   ", curr_user_vector, get_vector_mean(curr_user_vector))
        print("   ", similar_user_vector, get_vector_mean(similar_user_vector))

    print("\n item-item recommendations for user %d: \n" % curr_user,
          recommendations_by_items(data, curr_user, max_similar_items=7))
    print("\n base line recommendations for user %d: \n" % curr_user,
          recommendations_by_items(data, curr_user, max_similar_items=7, allow_base_line=True))
    print("\n RMSE for user %d: \n" % curr_user, get_RMSE(data, curr_user))

    print("\n user-user recommendations for user %d: \n" % curr_user,
          recommendations_by_users(data, curr_user, max_similar_users=10))

    N, M = data.shape
    K = 8
    P = np.random.rand(N, K)
    Q = np.random.rand(M, K)

    #matrix_factorization(rating_data, P, Q, K, steps=-1)
    P, Q, step, err = matrix_factorization(data, P, Q, K)
    R = np.dot(P, Q.T)
    np.set_printoptions(precision=3, suppress=True, linewidth=75)
    print("\n matrix factorization (err = %f, step = %d, rmse = %f): \n" % (err, step, matrix_rmse(data, R)), R)


if __name__ == '__main__':
    test_recommender()
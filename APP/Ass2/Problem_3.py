# This code is to solve question 3.
import numpy as np
import matplotlib.pyplot as plt


ALPHA_UPPER = 0.9
ALPHA_CENTRAL = 0.9


def summation(array):
    return sum(array)


def upper_limit(cl, alpha):
    if cl >= 1 - alpha:
        return True
    else:
        return False


def central_lower_limit(cl, alpha):
    if cl >= (1 - alpha) / 2:
        return True
    else:
        return False


def central_upper_limit(cl, alpha):
    if cl >= (1 - alpha) / 2:
        return True
    else:
        return False


def factorial_array(n_array):
    n_fact = []
    for i, n in enumerate(n_array):
        n_fact.append(np.math.factorial(n))
    return np.array(n_fact)


def poisson_function(n_array, mu=0.5, b=3.0):
    factorials = factorial_array(n_array)
    return (mu+b)**n_array*np.exp(-(mu+b))/factorials


def filter_max(n_array, b=3.0):
    n_array_max = []
    for i, n in enumerate(n_array):
        n_array_max.append(max(0, n-b))
    return np.array(n_array_max)


def r_value(n, mu_best, mu=0.5, b=3.0):
    p1 = poisson_function(n, mu, b)
    p2 = poisson_function(n, mu_best, b)
    return p1 / p2


def arrange_array(list_of_indices, candidate_array):
    shifted_array = np.zeros(len(candidate_array))
    for q, n in enumerate(candidate_array):
        shifted_array[list_of_indices[q]] = n
    return shifted_array


def create_pairs(first, last):
    blended = np.vstack((first, last))
    pairs = np.transpose(blended)
    return pairs


def poisson_mu(x, mu=0.5):
    # Upper Limit
    x_vs = []
    pdf = poisson_function(x, mu+3)
    for q, P in enumerate(pdf):
        cl = summation(pdf[:q+1])
        if upper_limit(cl, ALPHA_UPPER) is True:
            x_vs.append(x[q])
            break

    # Central Interval
    x_1 = []
    mu_vs_cl = []
    pdf = poisson_function(x, mu+3.0)
    for q, P in enumerate(pdf):
        cl = summation(pdf[:q+1])
        if central_lower_limit(cl, ALPHA_CENTRAL) is True:
            for r, l in enumerate(x[q+1:]):
                cl_u = summation(pdf[-(r+1):-1])
                if central_upper_limit(cl_u, ALPHA_CENTRAL) is True:
                    x_1.append(x[q])
                    x_1.append(x[-r])
                    mu_vs_cl.append(mu)
                    break
            break
    return x_vs, x_1


def check_upper_limit(pair, limit):
    if limit == []:
        return 'N'
    else:
        if limit[0] <= pair[0]:
            return 'Y'
        else:
            return 'N'


def check_central_limit(pair, limits):
    if len(limits) < 2:
        return 'N'
    else:
        if min(limits) <= pair[0] and pair[1] <= max(limits):
            return 'Y'
        else:
            return 'N'


plt.figure(figsize=(16, 9))
mu = 0
mu_vs = []
n_vs = []
while mu <= 20:
    n_values = np.arange(0, 20, 1)
    # # print(n_values)
    p_mu = poisson_function(n_values, mu)
    # # print(p_mu)
    mu_best = filter_max(n_values)
    # # print(mu_best)
    p_mu_best = poisson_function(n_values, mu_best)
    # # print(p_mu_best)
    r_values = r_value(n_values, mu_best, mu)
    # # print(r_values)
    # # # print(np.sort(r_values, reverse=True))
    sort_index = np.argsort(r_values)[::-1]
    rank = np.argsort(sort_index)+1
    # # print(rank)
    pairs = create_pairs(n_values, rank)
    # # print(pairs)

    upper_lt, central_lt = poisson_mu(n_values, mu)
    # # print(pairs)
    # # print(upper_lt, central_lt)
    store_n = []
    for q, n in enumerate(n_values):
        
        # # print('|', n, '|', p_mu[q], '|', mu_best[q], '|', p_mu_best[q], '|',
              # r_values[q], '|', rank[q], '|', pairs[q],
              # upper_lt, central_lt, check_upper_limit(pairs[q], upper_lt),
              # '|', check_central_limit(pairs[q], central_lt), '|')
        if check_upper_limit(pairs[q], upper_lt) == 'Y':
            # print(n, mu)
            # plt.plot(n, mu, 'o')
            store_n.append(n)
    if len(store_n) > 0:
        print(store_n)
        n_vs.append(store_n[0])
        mu_vs.append(mu)
    mu += 0.05
plt.plot(n_vs, mu_vs, 'o-')
plt.title('C.L. Upper Limit, Poissonian, b=3.0')
plt.xlabel('n')
plt.grid()
plt.ylabel('$\mu$')
plt.savefig('pr3_ul.png')


plt.figure(figsize=(16, 9))
mu = 0
mu_vs = []
n_vs_lower = []
n_vs_upper = []
while mu <= 20:
    n_values = np.arange(0, 20, 1)
    # # print(n_values)
    p_mu = poisson_function(n_values, mu)
    # # print(p_mu)
    mu_best = filter_max(n_values)
    # # print(mu_best)
    p_mu_best = poisson_function(n_values, mu_best)
    # # print(p_mu_best)
    r_values = r_value(n_values, mu_best, mu)
    # # print(r_values)
    # # # print(np.sort(r_values, reverse=True))
    sort_index = np.argsort(r_values)[::-1]
    rank = np.argsort(sort_index)+1
    # # print(rank)
    pairs = create_pairs(n_values, rank)
    # # print(pairs)

    upper_lt, central_lt = poisson_mu(n_values, mu)
    # # print(pairs)
    # # print(upper_lt, central_lt)
    store_n_lower = []
    store_n_upper = []
    for q, n in enumerate(n_values):
        
        # # print('|', n, '|', p_mu[q], '|', mu_best[q], '|', p_mu_best[q], '|',
              # r_values[q], '|', rank[q], '|', pairs[q],
              # upper_lt, central_lt, check_upper_limit(pairs[q], upper_lt),
              # '|', check_central_limit(pairs[q], central_lt), '|')
        if check_central_limit(pairs[q], central_lt) == 'Y':
            # print(n, mu)
            # plt.plot(n, mu, 'o')
            pairs[q].sort()
            store_n_lower.append(pairs[q][0])
            store_n_upper.append(pairs[q][1])
    if len(store_n_lower) > 0:
        print('lower', store_n_lower)
        n_vs_lower.append(store_n_lower[0])
        n_vs_upper.append(store_n_upper[0])
        mu_vs.append(mu)
    mu += 0.05
plt.plot(n_vs_lower, mu_vs, 'o-', label='lower')
plt.plot(n_vs_upper, mu_vs, 'o-', label='upper')
plt.title('C.L. Central Interval, Poissonian, b=3.0')
plt.xlabel('n')
plt.ylabel('$\mu$')
plt.grid()
plt.legend()
plt.savefig('pr3_cl.png')


# End



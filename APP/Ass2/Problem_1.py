# This code is to generate the Neyman construction for different distribution
# functions with central and upper confidence limits.

import numpy as np
import matplotlib.pyplot as plt


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def poisson_distribution(x, mu):
    func = ((mu**x)/factorial(x)) * np.exp(-mu)
    return func


def poisson_function(x_array, mu):
    func_values = []
    for n in x_array:
        func_values.append(poisson_distribution(n, mu))
    return np.array(func_values)


def summation(array):
    return sum(array)


def total_prob(x, pdf, x1):
    mask = x == x1
    index = np.where(mask)
    print(index)
    sum_to_x1 = summation(pdf[:index+1])
    return sum_to_x1


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



x = np.arange(0, 10, 1)

alpha_upper = 0.9
mu = 0

# Upper Limit
x_vs = []
mu_vs = []
while mu < 10:
    pdf = poisson_function(x, mu)
    for q, P in enumerate(pdf):
        cl = summation(pdf[:q+1])
        print(x[q], P, cl)
        if upper_limit(cl, alpha_upper) is True:
            print('we got it')
            x_vs.append(x[q])
            mu_vs.append(mu)
            break
    mu += 1

plt.figure()
plt.plot(x_vs, mu_vs, 'o-')
plt.title('Poisson, Upper limit 90%')
plt.xlabel('Measured x')
# plt.show(block=False)
plt.savefig('poisson_upper.png')

# Central Interval
print('Central Interval')
alpha_central = 0.68
x_1 = []
x_2 = []
mu_vs_cl = []
mu = 0
while mu < 10:
    pdf = poisson_function(x, mu)
    for q, P in enumerate(pdf):
        cl = summation(pdf[:q+1])
        print(x[q], P, cl)
        if central_lower_limit(cl, alpha_central) is True:
            print('we got lower it')

            for r, l in enumerate(x[q+1:]):
                print(r, q, r+q+1)
                r += q+1
                cl_u = summation(pdf[r:])
                if central_upper_limit(cl_u, alpha_central) is True:
                    x_2.append(x[r])
                    x_1.append(x[q])
                    mu_vs_cl.append(mu)
                    break
            break
    mu += 1
print(x_1)
print(x_2)
plt.figure()
plt.plot(x_1, mu_vs_cl, 'o-')
plt.plot(x_2, mu_vs_cl, 'o-')
plt.title('Poisson, central limit 68%')
plt.xlabel('Measured x')
# plt.show(block=False)
plt.savefig('poisson_central.png')

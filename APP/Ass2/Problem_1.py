# This code is to generate the Neyman construction for different distribution
# functions with central and upper confidence limits.

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

ALPHA_UPPER = 0.9
ALPHA_CENTRAL = 0.68


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


def gaussian_function(x_array, mu):
    return (1/(np.sqrt(2*np.pi))) * np.exp(-(x_array-mu)**2/2)


def integral_gaussian(x_0, x_u, mu):
    print(x_0, x_u)
    Integral = integrate.quad(gaussian_function, x_0, x_u, args=mu)
    return Integral[0]


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


def poisson():
    x = np.arange(0, 10, 1)
    mu = 0
    # Upper Limit
    x_vs = []
    mu_vs = []
    while mu < 10:
        pdf = poisson_function(x, mu)
        for q, P in enumerate(pdf):
            cl = summation(pdf[:q+1])
            print(x[q], P, cl)
            if upper_limit(cl, ALPHA_UPPER) is True:
                print('we got it')
                x_vs.append(x[q])
                mu_vs.append(mu)
                break
        mu += 1

    plt.figure()
    plt.plot(x_vs, mu_vs, 'o-')
    plt.title('Poisson, Upper limit 90%')
    plt.xlabel('Measured x')
    plt.ylabel('$\mu$')
    plt.grid()
    # plt.show(block=False)
    plt.savefig('poisson_upper.png')

    # Central Interval
    print('Central Interval')
    x_1 = []
    x_2 = []
    mu_vs_cl = []
    mu = 0
    while mu < 10:
        pdf = poisson_function(x, mu)
        for q, P in enumerate(pdf):
            cl = summation(pdf[:q+1])
            print(x[q], P, cl)
            if central_lower_limit(cl, ALPHA_CENTRAL) is True:
                print('we got lower it')

                for r, l in enumerate(x[q+1:]):
                    print(r, q, r+q+1)
                    cl_u = summation(pdf[-(r+1):-1])
                    if central_upper_limit(cl_u, ALPHA_CENTRAL) is True:
                        x_2.append(x[-r-1])
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
    plt.ylabel('$\mu$')
    plt.grid()
    # plt.show(block=False)
    plt.savefig('poisson_central.png')
    return 0


def gaussian():
    x = np.arange(0.1, 10, 0.1)
    mu = 0
    # Upper Limit
    x_vs = []
    mu_vs = []
    while mu < 10:
        pdf = gaussian_function(x, mu)
        for q, P in enumerate(pdf):
            cl = integral_gaussian(0, x[q], mu)
            print(x[q], P, cl)
            if upper_limit(cl, ALPHA_UPPER) is True:
                print('we got it')
                x_vs.append(x[q])
                mu_vs.append(mu)
                break
        mu += 1

    plt.figure()
    plt.plot(x_vs, mu_vs, 'o-')
    plt.title('Gaussian, Upper limit 90%')
    plt.xlabel('Measured x')
    plt.ylabel('$\mu$')
    plt.grid()
    # plt.show(block=False)
    plt.savefig('gaussian_upper.png')

    # Central Interval
    print('Central Interval')
    x_1 = []
    x_2 = []
    mu_vs_cl = []
    mu = 0
    while mu < 10:
        pdf = gaussian_function(x, mu)
        for q, P in enumerate(pdf):
            cl = integral_gaussian(0, x[q], mu)
            print(x[q], P, cl)
            if central_lower_limit(cl, ALPHA_CENTRAL) is True:
                print('we got lower it')

                for r, l in enumerate(x[q+1:]):
                    print(r, q)
                    cl_u = integral_gaussian(x[-r-1], x[-1], mu)
                    if central_upper_limit(cl_u, ALPHA_CENTRAL) is True:
                        x_2.append(x[-r-1])
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
    plt.title('Gaussian, central limit 68%')
    plt.xlabel('Measured x')
    plt.ylabel('$\mu$')
    plt.grid()
    # plt.show(block=False)
    plt.savefig('gaussian_central.png')
    return 0


def uniform():
    # Upper Limit
    mu = 1
    x_vs = []
    mu_vs = []
    while mu < 100:
        k = 2 * mu
        x = np.arange(0, k, 1)
        print(x)
        pdf = 1/k + np.zeros(len(x))
        plt.figure()
        for q, P in enumerate(pdf):
            cl = summation(pdf[:q+1])
            print(mu,'x', x[q], 'P', P, 'limit', cl)
            if upper_limit(cl, ALPHA_UPPER) is True:
                x_vs.append(x[q])
                print('we got it', x_vs)
                mu_vs.append(mu)
                break
        mu += 1
    plt.figure()
    plt.plot(x_vs, mu_vs, 'o-')
    plt.title('Uniform, Upper limit 90%')
    plt.xlabel('Measured x')
    plt.ylabel('$\mu$')
    plt.grid()
    # plt.show(block=False)
    plt.savefig('uniform_upper.png')

    # Central Interval
    # print('Central Interval')
    x_1 = []
    x_2 = []
    mu_vs_cl = []
    mu = 1
    while mu < 100:
        k = 2 * mu
        x = np.arange(0, k, 1)
        pdf = 1/k + np.zeros(len(x))
        for q, P in enumerate(pdf):
            cl = summation(pdf[:q+1])
            # print(x[q], P, cl)
            if central_lower_limit(cl, ALPHA_CENTRAL) is True:
               # print('we got lower it')
                for r, l in enumerate(x[q+1:]):
                #    print(r, q, r+q+1)
                    cl_u = summation(pdf[-(r+1):-1])
                    if central_upper_limit(cl_u, ALPHA_CENTRAL) is True:
                        print(cl, cl_u)
                        x_2.append(x[-r-1])
                        x_1.append(x[q])
                        mu_vs_cl.append(mu)
                        break
                break
        mu += 1
    # print(x_1)
    # print(x_2)
    plt.figure()
    plt.plot(x_1, mu_vs_cl, 'o-')
    plt.plot(x_2, mu_vs_cl, 'o-')
    plt.title('Uniform, central limit 68%')
    plt.xlabel('Measured x')
    plt.ylabel('$\mu$')
    plt.grid()
    # plt.show(block=False)
    plt.savefig('uniform_central.png')
    # return 0


uniform()
# gaussian()
# poisson()
# Code ends here

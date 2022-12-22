# This code is to generate the Neyman construction for different distribution
# functions with central and upper confidence limits.

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

ALPHA_UPPER = 0.9
ALPHA_CENTRAL = 0.68


def gaussian_function(x_array, mu, sigma):
    return (
        1/(sigma * np.sqrt(2*np.pi))
    ) * np.exp(-(x_array-mu)**2/(2*sigma**2))


def integral_gaussian(x_0, x_u, mu, sigma):
    print(x_0, x_u)
    Integral = integrate.quad(gaussian_function, x_0, x_u, args=(mu, sigma))
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


colormap = plt.get_cmap('jet')


def scale(i, mini=0, maxi=10):
    numb = int(255*(i-mini)/(maxi-mini))
    return numb


def gaussian():
    x = np.arange(0.1, 10, 0.1)
    sigma = 0.5
    plt.figure()
    while sigma < 5:
        mu = 0
        # Upper Limit
        x_vs = []
        mu_vs = []
        while mu < 10:
            pdf = gaussian_function(x, mu, sigma)
            for q, P in enumerate(pdf):
                cl = integral_gaussian(0, x[q], mu, sigma)
                print(x[q], P, cl)
                if upper_limit(cl, ALPHA_UPPER) is True:
                    print('we got it')
                    x_vs.append(x[q])
                    mu_vs.append(mu)
                    break
            mu += 1
        plt.plot(x_vs, mu_vs, 'o-', color=colormap(scale(sigma, 0, 5)),
                 label='$\sigma$ = {}'.format(sigma))

        sigma += 0.5

    plt.title('Gaussian, Upper limit 90%')
    plt.xlabel('Measured x')
    plt.ylabel('$\mu$')
    plt.grid()
    plt.legend()
    # plt.show(block=False)
    plt.savefig('gaussian_upper_diff_sigma.png')

    # Central Interval
    print('Central Interval')
    plt.figure()
    sigma = 0.5
    while sigma < 5:
        x_1 = []
        x_2 = []
        mu_vs_cl = []
        mu = 0
        while mu < 10:
            pdf = gaussian_function(x, mu, sigma)
            for q, P in enumerate(pdf):
                cl = integral_gaussian(0, x[q], mu, sigma)
                print(x[q], P, cl)
                if central_lower_limit(cl, ALPHA_CENTRAL) is True:
                    print('we got lower it')

                    for r, l in enumerate(x[q+1:]):
                        print(r, q)
                        cl_u = integral_gaussian(x[-r-1], x[-1], mu, sigma)
                        if central_upper_limit(cl_u, ALPHA_CENTRAL) is True:
                            x_2.append(x[-r-1])
                            x_1.append(x[q])
                            mu_vs_cl.append(mu)
                            break
                    break
            mu += 1
        print(x_1)
        print(x_2)

        plt.plot(x_1, mu_vs_cl, 'o-', color=colormap(scale(sigma, 0, 5)),
                 label='$\sigma$ = {}'.format(sigma))
        plt.plot(x_2, mu_vs_cl, 'o-', color=colormap(scale(sigma, 0, 5)))
        sigma += 0.5
    plt.title('Gaussian, central limit 68% with different values of $\sigma$ ')
    plt.xlabel('Measured x')
    plt.ylabel('$\mu$')
    plt.grid()
    plt.legend()
    # plt.show(block=False)
    plt.savefig('gaussian_central_diff_sigma.png')


gaussian()

# Code ends here

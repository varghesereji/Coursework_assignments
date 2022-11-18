# This code is to get the correlation between two comb functions
# Which is randomly distributed

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import random
from matplotlib.backends.backend_pdf import PdfPages


def dirac_deltas(N, n):
    '''
    Parameters
    -----------------------------
    N: Number of elements in the array
    n: Position of dirac delta function in the array
    '''
    Signal = signal.unit_impulse(N, n)
    return Signal


def random_dirac_delta(N):
    '''
    Parameters
    -------------------------------------
    N: Number of elements
    '''
    random_position = random.randint(0, N)
    delta_n = dirac_deltas(N, random_position)
    return delta_n


def combine_dirac_deltas(sum_array, new_array):
    updated_sum = sum_array + new_array
    return updated_sum


def pulse_train(no_pulses, N):
    pulse_number = 0
    sum_array = None
    while (pulse_number < no_pulses):
        delta_function = random_dirac_delta(N)
        mask = delta_function == 1
        if sum_array is None:
            sum_array = delta_function
        else:
            if sum_array[mask] == 0:
                sum_array = combine_dirac_deltas(sum_array, delta_function)
        pulse_number += 1
    return sum_array


def function_convolution(function_1, function_2):
    convolved_function = np.convolve(function_1, function_2)
    x_c = np.linspace(-5, 5, len(convolved_function))
    return [x_c, convolved_function]


def generate_gaussian_train(N=1000, n=100):
    gaussian_signal = signal.gaussian(n, 2, True)
    x_g = np.linspace(-1, 1, n)
    function_array = pulse_train(30, N)
    x_d = np.linspace(-5, 5, N)
    x_t, gaussian_train = function_convolution(gaussian_signal, function_array)
    return [x_g, gaussian_signal, x_d, function_array, x_t, gaussian_train]


pdfplots = PdfPages('gaussian_signal_correlation.pdf')

x_g, gaussian_signal, \
    x_d, function_array_1, \
    x_t, gaussian_train_1 \
    = generate_gaussian_train()

fig, axs = plt.subplots(3, figsize=(16, 9))
axs[0].plot(x_g, gaussian_signal)
axs[1].plot(x_d, function_array_1)
axs[2].plot(x_t, gaussian_train_1)
axs[0].set_title('First signal')
pdfplots.savefig()
plt.close()


x_g, gaussian_signal, \
    x_d, function_array_2, \
    x_t, gaussian_train_2 \
    = generate_gaussian_train()

fig, axs = plt.subplots(3, figsize=(16, 9))
axs[0].plot(x_g, gaussian_signal)
axs[1].plot(x_d, function_array_2)
axs[2].plot(x_t, gaussian_train_2)
axs[0].set_title('Second signal')
pdfplots.savefig()
plt.close()


x_f, train_correlation = function_convolution(gaussian_train_1,
                                              gaussian_train_2)
plt.figure(figsize=(16, 9))
plt.plot(x_f, train_correlation)
plt.title('Final correlation')
pdfplots.savefig()
plt.close()

pdfplots.close()

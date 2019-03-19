'''

How to use:

import lamdbata_ruwai.pyfuncs as pyf

pyf.kaggle_submission(predictions, X_test, destination)


'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def ecdf(data):

    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y

def bernoulli_distribution(n, p):

    count = 0

    for i in range(n):

        random_number = np.random.random()

        if random_number < p:

            count += 1

    return count

def plot_cdf(size=10000):

    np.random.seed(42)
    n_defaults = np.random.binomial(100, 0.05, size)

    x, y = ecdf(n_defaults)

    _ = plt.plot(x, y, marker='.', linestyle='none')
    plt.margins(0.002)
    plt.xlabel('Defaults out of 100 loans')
    plt.ylabel('ECDF')

    plt.show()

def bernoulli_plot():

    np.random.seed(42)
    n_defaults = np.empty(1000)

    for i in range(1000):
        n_defaults[i] = bernoulli_distribution(100, 0.05)

    bins = np.arange(-0.5, max(n_defaults + 1.5) - 0.5)

    plt.margins(0.02)

    _ = plt.hist(n_defaults, normed=True, bins=bins)
    _ = plt.xlabel('number of defaults out of 100 loans')
    _ = plt.ylabel('Binomial PMF')

    plt.show()

def poisson_distribution(size=10000,n=[20, 100, 1000],p=[0.5, 0.1, 0.01]):

    np.random.seed(42)

    samples_poisson = np.random.poisson(10, size)

    print('Poisson:     ', np.mean(samples_poisson),
                           np.std(samples_poisson))

    for i in range(3):
        samples_binomial = np.random.binomial(n[i], p[i], size)

        # Print results
        print('n =', n[i], 'Binomial:   ', np.mean(samples_binomial),
                                           np.std(samples_binomial))

    return samples_poisson

def create_kaggle_submission(predictions,X_test,destination):

    preds=pd.DataFrame(predictions,columns=['target'])

    ids=pd.DataFrame(X_test.id,columns=['id'])

    submission = pd.concat([ids, preds], axis=1)

    submission.to_csv(destination, index=False, header=['id','target'])

    return


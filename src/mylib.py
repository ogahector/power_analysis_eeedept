import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions

def print_data_statistics(data: pd.Series) -> None:
    print("DATASET STATISTICS:")
    print(f"Mean: {np.mean(data)}")
    print(f"Standard Deviation: {np.std(data)}")
    print(f"Minimum Value: {min(data)}")
    print(f"25th Percentile: {np.percentile(data, 25)}")
    print(f"Median: {np.median(data)}")
    print(f"75th Percentile: {np.percentile(data, 75)}")
    print(f"Maximum: {max(data)}")
    print(f"Skewness: {stats.skew(data)}")
    print(f"Kurtosis: {stats.kurtosis(data)}")
    print()
    print('STATISTICAL TESTS FOR NORMALITY:')
    print(f"Shapiro Wilk Test: stat={stats.shapiro(data)[0]:.4f}, p={stats.shapiro(data)[1]}")
    print(f'DAgostino and Pearsons Test: stat={stats.normaltest(data)[0]:.4f}, p={stats.normaltest(data)[1]}')
    ks_test = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    print(f'Kolmogorov-Smirnov Test: stat={ks_test[0]}, p={ks_test[1]}')
    print()


def fit_using_Fitter(data: pd.Series) -> None:
    fit = Fitter(data, distributions=['norm', 'lognorm', 'expon', 'gamma', 'beta'])
    # fit = Fitter(data)
    fit.fit()
    fit.summary(Nbest=5)
    print(f"Best Fit: {fit.get_best()}")


def get_best_dist(data: pd.Series) -> None: 
    dist_names = get_common_distributions()
    dist_names.extend(['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 
                          'dweibull', 't', 'pareto', 'exponnorm', 'lognorm',
                         "norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"])
    dist_results = []
    params = {}
    print('FITTING TO OTHER DISTRIBUTIONS: ')
    for dist_name in dist_names:
        dist = getattr(stats, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        _, p = stats.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("\nBest fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


def statistical_analysis(data: pd.Series, bins=15, header='Power', cdf=False) -> None:
    
    print_data_statistics(data)

    FIGDIM = (2, 2)

    ## HISTOGRAM AND KDE
    plt.figure(figsize=(16, 10))
    ax = []
    ax.append(plt.subplot(*FIGDIM, 1))
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.75, label='Histogram')
    plt.ylabel('Number of Samples')
    plt.xlabel(header)
    plt.legend(loc='upper left')
    ax[0]=ax[0].twinx()
    data.plot.kde(color='r')
    plt.legend(['KDE'], loc='upper right')
    plt.title(header+' Histogram and KDE')

    ## Q-Q PLOT
    ax.append(plt.subplot(*FIGDIM, 2))
    (_, (slope, intercept, _)) = stats.probplot(data, dist='norm', plot=plt, rvalue=True)
    plt.annotate(text=f'y={slope:.4f}*x + {intercept:.4f}', xy=(0.5, 0.15), xycoords='axes fraction', fontsize=16, color='r')
    plt.grid()
    plt.legend(['Q-Q', 'Normal Dist'], loc='upper left')
    plt.title('Q-Q Plot for Normal Distribution')

    ## FIT TO OTHER DISTRIBUTIONS / FIND BEST FIT
    ax.append(plt.subplot(*FIGDIM, 3))
    dist_name, p, dist_args = get_best_dist(data)
    dist = getattr(stats, dist_name)
    x = np.linspace(min(data), max(data), 100)
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.75, label='Histogram')
    plt.xlabel(header)
    plt.ylabel('Number of Samples')
    plt.legend(loc='upper left')

    ax[-1]=ax[-1].twinx()
    plt.plot(x, dist.pdf(x, *dist_args), color='r', linestyle='-', label='Dist of Best Fit PDF')
    if cdf: plt.plot(x, dist.cdf(x, *dist_args), color='g', linestyle='-', label='Dist of Best Fit CDF')
    plt.annotate(text=dist_name.upper(), xy=(0.7, 0.7), xycoords='axes fraction', fontsize=16, color='r')
    plt.legend(loc='upper right')

    ## Q-Q PLOT FOR THE BEST FIT
    ax.append(plt.subplot(*FIGDIM, 4))
    (_, (slope, intercept, _)) = stats.probplot(data, sparams=dist_args, dist=dist_name, fit=True, plot=plt, rvalue=True)
    plt.annotate(text=f'y={slope:.4f}*x + {intercept:.4f}', xy=(0.5, 0.15), xycoords='axes fraction', fontsize=16, color='r')
    plt.grid()
    plt.legend(['Q-Q', dist_name.upper()+' Dist'])
    plt.show()

# if __name__ == '__main__':
    # main()
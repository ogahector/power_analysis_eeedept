import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from fitter import Fitter

print('testingsuite is loaded')
print(__file__)

def print_statistical_properties(data: pd.Series) -> None:
    print("DATASET STATISTICS:")
    print("Mean: "+str(np.mean(data)))
    print("Standard Deviation: "+str(np.std(data)))
    print("Minimum Value: "+str(min(data)))
    print("25th Percentile: "+str(np.percentile(data, 25)))
    print("Median: "+str(np.median(data)))
    print("75th Percentile: "+str(np.percentile(data, 75)))
    print("Maximum: "+str(max(data)))
    print()
    print("Skewness: "+str(stats.skew(data)))
    print("Kurtosis: "+str(stats.kurtosis(data)))
    print()
    print('STATISTICAL TESTS FOR NORMALITY:')
    print(f"Shapiro Wilk Test: stat={stats.shapiro(data)[0]:.4f}, p={stats.shapiro(data)[1]}")
    print(f'DAgostino and Pearsons Test: stat={stats.normaltest(data)[0]:.4f}, p={stats.normaltest(data)[1]}')
    ks_test = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    print(f'Kolmogorov-Smirnov Test: stat= {ks_test[0]}, p= {ks_test[1]}')


# this function 
def statistical_analysis(data: pd.Series, bins=15, header='Power') -> None:
    print_statistical_properties(data)

    FIGDIM = (2, 2)

    ## HISTOGRAM AND KDE
    plt.figure(figsize=(16, 10))
    ax = []
    ax.append(plt.subplot(*FIGDIM, 1))
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.75, label='Histogram')
    plt.legend(loc='upper left')
    ax[0]=ax[0].twinx()
    data.plot.kde(color='r')
    plt.legend(['KDE'], loc='upper right')
    plt.title(header+'Histogram and KDE')

    ## Q-Q PLOT
    ax.append(plt.subplot(*FIGDIM, 2))
    sm.qqplot(data, fit=False, line='r', ax=ax[-1])
    plt.grid()
    plt.legend(['Q-Q', 'Normal Dist'], loc='upper left')
    plt.title('Q-Q Plot for Normal Distribution')

    ## GOODNESS-OF-FIT TEST

    ## FIT TO OTHER DISTRIBUTIONS


    plt.show()

# if __name__ == '__main__':
#     analyse(pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
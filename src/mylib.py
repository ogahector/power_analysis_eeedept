import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.graphics.gofplots as sm
import matplotlib.pyplot as plt
from fitter import Fitter, get_common_distributions
from seaborn import kdeplot

## IMPORT DATA
def import_csv(filename:str='') -> pd.DataFrame:
    df = pd.read_csv(filename).drop(columns="No.").drop(0)
    df["Date/Time"] = pd.to_datetime(df["Date/Time"], format='%d/%m/%Y %H:%M', utc=True)
    df.set_index(['Date/Time'], inplace=True, drop=False)
    return df


def import_cpu_usage(filename:str='', ref:pd.DataFrame|None=None, truncate:bool=True) -> pd.DataFrame:
    df = pd.read_csv(filename)
    df['now'] = pd.to_datetime(df['now'], utc=True)
    if truncate: df = df.set_index(df['now'], drop=False).truncate(before=ref.index[0], 
                                                  after=ref.index[-1])
    return df

### STATISTICAL ANALYSIS
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


def fit_using_Fitter(data: pd.Series) -> None: # TAKES MASSIVE AMOUNTS OF TIME: DO NOT USE!
    dist_names = get_common_distributions()
    dist_names.extend(['arcsine', 'cosine', 'expon', 'weibull_max', 'weibull_min', 
                          'dweibull', 't', 'pareto', 'exponnorm', 'lognorm',
                         "norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"])
    fit = Fitter(data, distributions=dist_names)
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
    # store the name of the best fit and its p value
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))

    print("\nBest fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


def statistical_analysis(data: pd.Series, bins=15, header='Power', cdf=False, modes:list=[]) -> None:
    
    print_data_statistics(data)

    FIGDIM = (2, 2)

    # GET BEST DISTS AND DISTARGS
    dist_name, p, dist_args = get_best_dist(data)
    dist = getattr(stats, dist_name)
    x = np.linspace(min(data), max(data), 100)

    ## HISTOGRAM AND KDE
    plt.figure(figsize=(16, 10))
    ax = []
    ax.append(plt.subplot(*FIGDIM, 1))
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.75, label='Histogram')
    plt.vlines(modes, ymin=ax[-1].get_ylim()[0], ymax=ax[-1].get_ylim()[1], colors='g', linewidth=3)
    plt.ylabel('Number of Samples')
    plt.xlabel(header)
    plt.legend(loc='upper left')
    ax[0]=ax[0].twinx()
    # data.plot.kde(color='r')
    kdeplot(data, color='r')
    plt.legend(['KDE'], loc='upper right')
    plt.title(header+' Histogram and KDE')

    ## Q-Q PLOT
    ax.append(plt.subplot(*FIGDIM, 2))
    # plt.annotate(text=f'y={slope:.4f}*x + {intercept:.4f}', xy=(0.5, 0.15), xycoords='axes fraction', fontsize=16, color='r')
    sm.qqplot(data=data, dist=stats.norm, distargs=(), loc=np.mean(data), scale=np.std(data), fit=True, line='45', ax=ax[-1])
    plt.grid()
    plt.legend(['Q-Q', 'Normal Dist'], loc='upper left')
    plt.title('Q-Q Plot for Normal Distribution')

    ## FIT TO OTHER DISTRIBUTIONS / FIND BEST FIT
    ax.append(plt.subplot(*FIGDIM, 3))
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.75, label='Histogram')
    plt.vlines(modes, ymin=ax[-1].get_ylim()[0], ymax=ax[-1].get_ylim()[1], colors='g', linewidth=3)
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
    # plt.annotate(text=f'y={slope:.4f}*x + {intercept:.4f}', xy=(0.5, 0.15), xycoords='axes fraction', fontsize=16, color='r')
    sm.qqplot(data=data, dist=dist, distargs=(), loc=dist_args[0], scale=dist_args[1], fit=True, line='45', ax=ax[-1])
    plt.grid()
    plt.legend(['Q-Q', dist_name.upper()+' Dist'])
    plt.show()


## HELPER AND UTILITIES
def hour_to_Timedelta(h: any, unit:str='h') -> pd.Timedelta | list[pd.Timedelta]:
    if type(h) == float: return pd.Timedelta(h, unit=unit)
    elif hasattr(h, '__len__'): return [pd.Timedelta(i, unit=unit) for i in h]
    else: return None


def separate_by_weekdays(data:pd.Series, header:str='') -> pd.Series:
    days_of_the_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    print("Mean total " + header + ": " + str(data.mean()) + " kW")
    print("Median total " + header + ": " + str(data.median()) + " kW\n")

    ret = data.groupby(data.index.weekday).apply(list)

    ret["Weekend"] = ret[5] + ret[6] # 5 == Saturday, 6 == Sunday
    ret["Weekdays"] = ret[0] + ret[1] + ret[2] + ret[3] + ret[4]

    print("Mean Weekend " + header + ": " + str(np.mean(ret['Weekend'])))
    print("Mean Weekdays " + header + ": " + str(np.mean(ret['Weekdays'])))

    for i in range(7):
        print("Mean " + header + " on " + days_of_the_week[i] + ": " + str(np.mean(ret[i])))
    for i in range(7):
        print("Median " + header + " on " + days_of_the_week[i] + ": " + str(np.median(ret[i])))

    return ret


def cpu_corr_by_ref(cpu:pd.Series, ref:pd.Series, time:str, passedin:dict, name:str='', normalize:bool=False) -> None:
    # modifies IN PLACE
    if normalize: 
        passedin[time] = cpu.groupby(cpu.index.normalize()).mean().truncate(
        before=ref.index[0], after=ref.index[-1])
    else: 
        passedin[time] = cpu.groupby(cpu.index.round(time)).mean().truncate(
        before=ref.index[0], after=ref.index[-1])
    print(f'{name} {time} Correlation: {ref.corr(passedin[time])}')
    

## PLOTS
def plot_raw_data(df:pd.DataFrame, figsize:tuple=(16, 5)) -> None:
    headers = df.columns.values.tolist()[1:]

    figsize = (figsize[0], figsize[1]*(len(headers)))

    plt.figure(figsize=figsize)

    for i, name in enumerate(headers):
        plt.subplot(len(headers), 1, i+1)
        plt.plot(df.index, df[name], '-b')
        plt.xlabel('Date/Time')
        plt.ylabel(name)
        plt.title(name + ' against Date/Time')

    plt.tight_layout()
    plt.show()
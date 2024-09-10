"""
This is Hector Oga's Library
For the Summer 2024 UROP on Server Room 1005 Power Analysis 
Find here all the functions used to streamline 
The data analysis process
13/09/24
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as sig
import scipy.fft as fft
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
        # print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    # store the name of the best fit and its p value
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))

    print("\nBest fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]


def statistical_analysis(data: pd.Series, bins=15, header='Power', cdf=False, modes:list=[], include_autocorr:bool=False) -> None:
    print(header.upper().center(75, '-'))
    print_data_statistics(data)

    FIGDIM = (3, 2) if include_autocorr else (2, 2)

    # GET BEST DISTS AND DISTARGS
    dist_name, p, dist_args = get_best_dist(data)
    dist = getattr(stats, dist_name)
    x = np.linspace(min(data), max(data), 100)

    ## HISTOGRAM AND KDE
    plt.figure(figsize=(16, 10))
    ax = []
    ax.append(plt.subplot(*FIGDIM, 1))
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.75, label='Histogram')
    ylims = plt.gca().get_ylim()
    plt.vlines(modes, ymin=ylims[0], ymax=ylims[1], colors='g', linewidth=3)
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
    ylims = plt.gca().get_ylim()
    plt.vlines(modes, ymin=ylims[0], ymax=ylims[1], colors='g', linewidth=3)
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

    if include_autocorr:
        ## AUTOCORRELATION
        ax.append(plt.subplot(*FIGDIM, 5))
        pd.plotting.autocorrelation_plot(data, ax=plt.gca(), color='b', linestyle='-', label='Autocorrelation')
        plt.plot(data.autocorr(), 'xr', label='Autocorrelation for Lag 1')
        plt.grid()
        plt.legend()

        ## PSD
        ax.append(plt.subplot(*FIGDIM, 6))
        fs = df_av_sampling_freq(data)
        freqs, psd = sig.welch(data, fs)
        plt.semilogy(freqs, psd, '-b', label='PSD')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power/Frequency (dB/Hz)')
        plt.grid()
    
    plt.show()


def print_psd_stats(df:pd.Series, freqs, psd, psd_peaks) -> None:
    print(f'Autocorrelation for Lag 1: {df.autocorr()}')
    print(f'Maximum Autocorrelation (dB): {max(psd)}')
    print(f'Times where a peak occurs: {hour_to_Timedelta(1/(3600*freqs[psd_peaks]))}')
    print(f'Peak Amplitudes: {[peak for peak in psd[psd_peaks]]}')


def autocorr_and_psd_analysis(data: pd.Series, header='Power', peaks_height:float=0, double_sided:bool=False) -> list | None:
    fs = df_av_sampling_freq(data)
    freqs, psd = sig.welch(data, fs, return_onesided=not double_sided)
    psd = 10*np.log10(psd)
    psd_peaks, _ = sig.find_peaks(psd, height=peaks_height)

    periods = hour_to_Timedelta(1/(3600*freqs[psd_peaks]))

    print_psd_stats(data, freqs, psd, psd_peaks)

    FIGDIM = (2, 2)
    plt.figure(figsize=(16, 10))
    ax = []

    ## AUTOCORRELATION
    ax.append(plt.subplot(*FIGDIM, 1))
    pd.plotting.autocorrelation_plot(data, ax=plt.gca(), color='b', linestyle='-', label=header+' Autocorrelation')
    plt.plot(data.autocorr(), 'xr', label=header+' Autocorrelation for Lag 1')
    plt.grid()
    plt.legend()

    ## PSD
    ax.append(plt.subplot(*FIGDIM, 2))
    if double_sided: 
        plt.plot(fft.fftshift(freqs)*3600, fft.fftshift(psd), '-b', label=header+' PSD')
    else: 
        plt.plot(freqs*3600, psd, '-b', label=header+' PSD')
    plt.plot(freqs[psd_peaks]*3600, psd[psd_peaks], 'xr', label='Main PSD Peaks')
    plt.legend(loc='upper right')
    plt.xlabel('Frequency (1/hour)')
    plt.ylabel('Power/Frequency (dB)')
    plt.grid()

    ## PLOT MOST IMPORTANT PERIOD
    ax.append(plt.subplot(*FIGDIM, 3))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    periods = periods[:2]
    idx = np.multiply(Timedelta_to_hour(periods), fs*3600)
    idx = [int(x) for x in idx]

    plt.plot(data.index, data, label='Main Data')
    plt.plot(data.index[::idx[0]], data[::idx[0]], colors[0]+'x', label=str(periods[0]))
    plt.legend(loc='upper right')
    plt.xlabel('Date/Time')
    plt.ylabel(header)
    plt.title(header + f' against Date/Time with main period marked')

    ## PLOT SECOND MOST IMPORTANT PERIOD
    ax.append(plt.subplot(*FIGDIM, 4))
    plt.plot(data.index, data, label='Main Data')
    plt.plot(data.index[::idx[1]], data[::idx[1]], colors[1]+'x', label=str(periods[1]))
    plt.legend(loc='upper right')
    plt.xlabel('Date/Time')
    plt.ylabel(header)
    plt.title(header + f' against Date/Time with 2nd main period marked')

    plt.tight_layout()
    plt.show()

    return periods


## HELPER AND UTILITIES
def hour_to_Timedelta(h: float | np.float_ | list[float] | list[np.float_], unit:str='h') -> pd.Timedelta | list[pd.Timedelta]:
    if hasattr(h, '__len__') and type(h) != str: return [hour_to_Timedelta(i) for i in h]
    elif type(h) == float or type(h) == np.float_: return pd.Timedelta(h, unit=unit)
    else: raise TypeError('Unexpected Input to Function hour_to_Timedelta!')


def Timedelta_to_hour(t:pd.Timedelta | list[pd.Timedelta] | str) -> float | list[float]:
    if hasattr(t, '__len__') and type(t) != str and type(t) != pd.Timedelta: return [Timedelta_to_hour(val) for val in t]
    elif type(t) == pd.Timedelta: return t.total_seconds()/3600
    else: return pd.Timedelta(t).total_seconds()/3600


def find_nearest(array:any, val:any=0) -> any:
    if hasattr(val, '__len__'): return [find_nearest(array, i) for i in val]
    diff_idx = np.abs(array - val).argmin()
    return array[diff_idx] if type(array) != pd.Series else array.iloc[diff_idx], diff_idx


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
    

def get_name(my_var:any) -> str:
    return [name for name, v in locals().items() if v is my_var or v == my_var]


def df_av_sampling_time(df:pd.DataFrame | pd.Series) -> float:
    return pd.Series(df.index).diff().dropna().apply(Timedelta_to_hour).mean()*3600

def df_av_sampling_freq(df:pd.DataFrame | pd.Series) -> float:
    return 1.0/df_av_sampling_time(df)

## PLOTS
def plot_raw_data(df:pd.DataFrame | list[pd.DataFrame], figsize:tuple=(16, 5), columnname:str='') -> None:
    if type(df) == pd.DataFrame:
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

    elif type(df) == list:
        l = len(df)
        fig, ax = plt.subplots(4, figsize=(figsize[0], l*figsize[1]))

        for i, data in enumerate(df):
            ax[i].plot(data.index, data[columnname], color='b', linestyle='-')
            ax[i].set_xlabel('Date/Time')
            ax[i].set_ylabel(f'DataFrame {i} Server CPU Usage')
            ax[i].set_ylim(0, 100)
            ax[i].set_xlim(data.index[0], data.index[-1])
        ax[0].set_title("All CPU Usages in Time")
        plt.tight_layout()
        plt.show()

def plot_hourly_av(data:pd.Series, header:str='', figsize:tuple=(16, 5)) -> None:
    hourly_av = data.groupby(data.index.hour).apply(list).mean()
    hours = np.arange(24)

    plt.figure(figsize=figsize)
    plt.plot(hours, hourly_av, '-b', label=header+' Hourly Av')
    plt.xticks(hours, [str(i) for i in hours])
    plt.xlabel('Date/Time')
    plt.ylabel(header)
    plt.title(header+' Hourly Average Throughout a Day')
    plt.show()


def plot_weekly_hourly_av(data:pd.Series, header:str='', figsize:tuple=(16, 5), serially:bool=False) -> None:
    daily_hourly_av = data.groupby([data.index.weekday, data.index.hour]).apply(list).apply(np.mean)
    hours = np.arange(24)
    days = np.arange(7)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    days_of_the_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    time_scaler = (days)*24 if serially else np.zeros(len(days))

    plt.figure(figsize=figsize)
    for i, color in enumerate(colors):
        plt.plot(hours+time_scaler[i], daily_hourly_av[i], 
                    color=color, linestyle='-', marker='o', label=f'{header} on {days_of_the_week[i]}')
    if serially:
        plt.xticks(range(0, 24*7, 24), days_of_the_week)
    else:
        plt.xticks(hours, [str(hour) for hour in hours])
    if not serially: plt.legend(loc='upper right')
    plt.xlabel('Weekly Hourly Average')
    plt.ylabel(header)
    plt.grid()
    plt.title(header+' throughout the day on the average week')
    plt.show()

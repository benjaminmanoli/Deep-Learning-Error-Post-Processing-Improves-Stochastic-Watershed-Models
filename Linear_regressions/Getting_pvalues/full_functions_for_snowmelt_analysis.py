import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-v0_8-white')
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import os
plt.ioff()

########## GATHERING UP ERRORS OR LAMBDAS ##########
def gather_errors_or_lambdas(month, model, error_type, snow_years, nosnow_years, site):
    if model=='PRMS':
        if error_type == 'errors':
            error_df = pd.read_csv('calibrated_error_between_PRMS_and_observed.csv')
        if error_type == 'lambdas':
            error_df = pd.read_csv('lambda_PRMS.csv')
    if model=='HBV':
        if error_type == 'errors':
            error_df = pd.read_csv('error_between_HBV_and_observed.csv')
        if error_type == 'lambdas':
            error_df = pd.read_csv('lambda_HBV.csv')
    error_df['Date'] = pd.to_datetime(error_df['Date'])
    error_df['Year'] = error_df['Date'].dt.year
    error_df['Month'] = error_df['Date'].dt.month
    
    month_snow_error_array = np.array([])
    month_nosnow_error_array = np.array([])

    # these are the snowy years
    for year in snow_years:
        month_errors = error_df[(error_df['Year']==year) & (error_df['Month']==month)][site].dropna().values
        month_snow_error_array = np.append(month_snow_error_array, month_errors)

    # these are the non-snowy years
    for year in nosnow_years:
        month_errors = error_df[(error_df['Year']==year) & (error_df['Month']==month)][site].dropna().values
        month_nosnow_error_array = np.append(month_nosnow_error_array, month_errors)
    
    return month_snow_error_array, month_nosnow_error_array

########## PLOTTING HISTOGRAM AND KDE ##########
def plot_KDE_and_hist(snow_errors, nosnow_errors, error_type, month, model_type, site, lower_xlim, upper_xlim, binsize, smoothness):
    # KDE
    plt.figure(figsize=(10,8))
    sns.kdeplot(snow_errors, label='Snowy years', color='#4393c3', bw_adjust=smoothness)
    sns.kdeplot(nosnow_errors, label='Non-snowy years',color='#d6604d', bw_adjust=smoothness)
    plt.xlabel(error_type, fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.title('KDE of '+month+' '+model_type+' '+error_type+'\'s at site '+site, fontsize=15)
    plt.xlim(lower_xlim,upper_xlim)
    plt.legend(fontsize=13)
    plt.show()
    
    # Hist with different bins
    plt.figure(figsize=(10,8))
    sns.histplot(snow_errors, label='Snowy years', kde=True, stat='probability', color='#4393c3', alpha=0.5, 
                 kde_kws={'bw_adjust':smoothness})
    sns.histplot(nosnow_errors, label='Non-snowy years', kde=True, stat='probability', color='#d6604d', alpha=0.8, 
                 kde_kws={'bw_adjust':smoothness})
    plt.xlabel(error_type, fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.title('Histogram of '+month+' '+model_type+' '+error_type+'\'s at site '+site, fontsize=15)
    plt.xlim(lower_xlim,upper_xlim)
    plt.legend(fontsize=13)
    plt.show()
    
    # Hist with same bins
    plt.figure(figsize=(10,8))
    sns.histplot(snow_errors, label='Snowy years', kde=True, stat='probability', color='#4393c3', alpha=0.5, 
                 binwidth=binsize, kde_kws={'bw_adjust':smoothness})
    sns.histplot(nosnow_errors, label='Non-snowy years', kde=True, stat='probability', color='#d6604d', alpha=0.8, 
                 binwidth=binsize, kde_kws={'bw_adjust':smoothness})
    plt.xlabel(error_type, fontsize=13)
    plt.ylabel('Density', fontsize=13)
    plt.title('Histogram of '+month+' '+model_type+' '+error_type+'\'s at site '+site, fontsize=15)
    plt.xlim(lower_xlim,upper_xlim)
    plt.legend(fontsize=13)
    plt.show()
    
########## PLOTTING CDF ##########
def plot_cdf(snow_errors, nosnow_errors, error_type, month, model_type, site, lower_xlim, upper_xlim):
    plt.figure(figsize=(10,8))

    sns.ecdfplot(snow_errors, label='Snowy years', stat='proportion', color='#4393c3')
    sns.ecdfplot(nosnow_errors, label='Non-snowy years', stat='proportion', color='#d6604d')

    plt.xlabel(error_type, fontsize=13)
    plt.ylabel('Proportion', fontsize=13)
    plt.title('CDF of '+month+' '+model_type+' '+error_type+'\'s at site '+site, fontsize=15)

    plt.xlim(lower_xlim,upper_xlim)
    plt.legend(fontsize=13)
    plt.show()

########## FINDING AND PRINTING L-MOMENTS ##########
def print_Lmoments(X):
    X = X[~np.isnan(X)]
    n = len(X)
    X = np.sort(X)
            
    b0 = np.mean(X)
            
    run_sum = 0
    for i in range(1,n):
        run_sum = run_sum + (i+1-1)*X[i]
    b1 = run_sum/(n*(n-1))
            
    run_sum = 0
    for i in range(2,n):
        run_sum = run_sum + (i+1-1)*(i+1-2)*X[i]
    b2 = run_sum/(n*(n-1)*(n-2))
            
    run_sum = 0
    for i in range(3,n):
        run_sum = run_sum + (i+1-1)*(i+1-2)*(i+1-3)*X[i]
    b3 = run_sum/(n*(n-1)*(n-2)*(n-3))
            
    run_sum = 0
    for i in range(4,n):
        run_sum = run_sum + (i+1-1)*(i+1-2)*(i+1-3)*(i+1-4)*X[i]
    b4 = run_sum/(n*(n-1)*(n-2)*(n-3)*(n-4))
            
    l1 = b0
    l2 = 2*b1-b0
    l3 = 6*b2-6*b1+b0
    l4 = 20*b3-30*b2+12*b1-b0
    LSK = l3/l2
    LKU = l4/l2
    
    print(f'Mean: {l1:.2g}')
    print(f'L-scale: {l2:.2g}')
    print(f'L-skew: {LSK:.2g}')
    print(f'L-kurtosis: {LKU:.2g}')
    print(f'L-CV: {l2/l1:.2g}')
    
########## SCATTERPLOT OF SNOWMELT VS RUNOFF ##########
def plot_scatterplot(runoff_sum_df, snowmelt_sum_df, site, month):
    # need to trim this one since there's one more year included here
    runoff_sum_df = runoff_sum_df.loc[1916:]
    
    runoff_array = runoff_sum_df[site].values
    snowmelt_array = snowmelt_sum_df[site].values
    
    mask = ~np.isnan(runoff_array) & ~np.isnan(snowmelt_array)
    runoff_array = runoff_array[mask]
    snowmelt_array = snowmelt_array[mask]
    
    # plot
    fig = plt.figure(figsize=(10,8))
    plt.scatter(runoff_array, snowmelt_array, color='k')
    
    corr_coef = np.corrcoef(runoff_array, snowmelt_array)[0, 1]
    plt.annotate('Correlation coefficient: '+str(round(corr_coef,2)), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=13,
                 ha='left', va='top')
    
    plt.xlabel('Cumulative runoff in '+month, size=13)
    plt.ylabel('Cumulative snowmelt in '+month, size=13)
    plt.title('Snowmelt vs Runoff in '+month+' at site '+site, size=15)
    
    return fig
    
########## AUTOCORRELATION ##########
def plot_acf(snow_error_array, nosnow_error_array, acf_or_pacf, model, error_type, site):
    if acf_or_pacf == 'acf':
        snow_acf, snow_conf_int = sm.tsa.stattools.acf(snow_error_array, nlags=45, alpha=0.05)
    elif acf_or_pacf == 'pacf':
        snow_acf, snow_conf_int = sm.tsa.stattools.pacf(snow_error_array, nlags=45, alpha=0.05)
    snow_conf_int = snow_conf_int - np.column_stack((snow_acf, snow_acf))
    snow_conf_int = snow_conf_int[:,1]
    snow_lags = np.arange(len(snow_acf))
    
    if acf_or_pacf == 'acf':
        nosnow_acf, nosnow_conf_int = sm.tsa.stattools.acf(nosnow_error_array, nlags=45, alpha=0.05)
    elif acf_or_pacf == 'pacf':
        nosnow_acf, nosnow_conf_int = sm.tsa.stattools.pacf(nosnow_error_array, nlags=45, alpha=0.05)
    nosnow_conf_int = nosnow_conf_int - np.column_stack((nosnow_acf, nosnow_acf))
    nosnow_conf_int = nosnow_conf_int[:,1]
    nosnow_lags = np.arange(len(nosnow_acf))
    
    snow_lag_5 = snow_acf[5]
    snow_lag_10 = snow_acf[10]
    snow_lag_20 = snow_acf[20]
    
    nosnow_lag_5 = nosnow_acf[5]
    nosnow_lag_10 = nosnow_acf[10]
    nosnow_lag_20 = nosnow_acf[20]
    
    lags_array = [snow_lag_5, snow_lag_10, snow_lag_20, nosnow_lag_5, nosnow_lag_10, nosnow_lag_20]
    

    
    fig = plt.figure(figsize=(10,8))

    plt.plot(snow_lags, snow_acf, marker='o', color='#4393c3', label='Snowy years')
    plt.plot(nosnow_lags, nosnow_acf, marker='o', color='#d6604d', label='Non-snowy years')

    plt.fill_between(snow_lags, -snow_conf_int, snow_conf_int, alpha=0.25, color='#4393c3')
    plt.fill_between(nosnow_lags, -nosnow_conf_int, nosnow_conf_int, alpha=0.25, color='#d6604d')
    
    # plot the last point above (or below) the confidence interval
    snow_first_one_in_CI = np.where((snow_acf < snow_conf_int) & (snow_acf > -snow_conf_int))[0][0]
    nosnow_first_one_in_CI = np.where((nosnow_acf < nosnow_conf_int) & (nosnow_acf > -nosnow_conf_int))[0][0]
    
    plt.stem(snow_lags[snow_first_one_in_CI-1], snow_acf[snow_first_one_in_CI-1], linefmt='#4393c3', markerfmt='#4393c3', 
            basefmt='#4393c3')
    plt.stem(nosnow_lags[nosnow_first_one_in_CI-1], nosnow_acf[nosnow_first_one_in_CI-1], linefmt='#d6604d', 
                 markerfmt='#d6604d', basefmt='#d6604d')

    # label the last point above (or below) the confidence interval
    if snow_acf[snow_first_one_in_CI-1] > 0:
        snow_offset=0.02
    else:
        snow_offset=-0.04
    if nosnow_acf[nosnow_first_one_in_CI-1] > 0:
        nosnow_offset=0.02
    else:
        nosnow_offset=-0.04
        
    plt.text(snow_lags[snow_first_one_in_CI-1], snow_acf[snow_first_one_in_CI-1]+snow_offset, 
             str(snow_lags[snow_first_one_in_CI-1]), ha='left', fontsize=13, color='#4393c3')    
    plt.text(nosnow_lags[nosnow_first_one_in_CI-1],nosnow_acf[nosnow_first_one_in_CI-1]+nosnow_offset, 
             str(nosnow_lags[nosnow_first_one_in_CI-1]), ha='left', fontsize=13, color='#d6604d')

    plt.axhline(0, color='k')

    plt.legend(fontsize=13)

    plt.xlabel('Lag', size=13)
    if acf_or_pacf == 'acf':
        plt.ylabel('Autocorrelation function', size=13)
        plt.title('Autocorrelation of '+model+' '+error_type+'\'s in March and April at site '+site, size=15)
    if acf_or_pacf == 'pacf':
        plt.ylabel('Partial autocorrelation function', size=13)
        plt.title('Partial autocorrelation of '+model+' '+error_type+'\'s in March and April at site '+site, size=15)

    
    
    return lags_array, fig

########## PRMS L-MOMENTS FULL DATASET ##########
def PRMS_create_monthly_Lmoment_dfs(error_type, month1, month2):
    # load in the correct DF based on the error type
    if error_type == 'differenced errors':
        PRMS_errors_df = pd.read_csv('calibrated_error_between_PRMS_and_observed.csv')
    elif error_type == 'lambdas':
        PRMS_errors_df = pd.read_csv('lambda_PRMS.csv')
        
    # set up the error DF
    PRMS_errors_df['Date'] = pd.to_datetime(PRMS_errors_df['Date'])
    PRMS_errors_df['Year'] = PRMS_errors_df['Date'].dt.year
    PRMS_errors_df['Month'] = PRMS_errors_df['Date'].dt.month
    
    # set up dummy DFs for all the L-moments we want (for some reason for loops aren't working)
    # each DF just has years from 1915-2015 in the index, and each column name is a site number
    # these entries are all -99 for now, but they will eventually hold whatever L-moment is specified in the DF name
    
    #years = pd.Series(range(1915, 2016), name='Year')
    years = pd.Series(range(1951, 2016), name='Year')
    
    PRMS_month_errors_mean_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), columns=PRMS_errors_df.columns[1:-2])],
                                          axis=1)
    PRMS_month_errors_mean_df.set_index('Year', inplace=True)

    PRMS_month_errors_Lscale_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), columns=PRMS_errors_df.columns[1:-2])],
                                          axis=1)
    PRMS_month_errors_Lscale_df.set_index('Year', inplace=True)

    PRMS_month_errors_Lskew_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), columns=PRMS_errors_df.columns[1:-2])],
                                          axis=1)
    PRMS_month_errors_Lskew_df.set_index('Year', inplace=True)

    PRMS_month_errors_Lkurt_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), columns=PRMS_errors_df.columns[1:-2])],
                                          axis=1)
    PRMS_month_errors_Lkurt_df.set_index('Year', inplace=True)
    
    # choose the month of interest
    PRMS_month_errors_df = PRMS_errors_df[(PRMS_errors_df['Month']==month1) | (PRMS_errors_df['Month']==month2)]
    # get L-moment values for each year
    for year in years:
        PRMS_month_year_errors_df = PRMS_month_errors_df[PRMS_month_errors_df['Year']==year]
        # get L-moment values for each site
        for site in PRMS_month_year_errors_df.columns[1:-2]:
            # only calculate L-moments if the site has over 80% of available data in that given month
            if len(PRMS_month_year_errors_df[site].dropna()) >= 0.8*len(PRMS_month_year_errors_df):
                # now calculate L-moments of this array
                X = np.array(PRMS_month_year_errors_df[site].dropna())
                n = len(X)
                X = np.sort(X)
            
                b0 = np.mean(X)
            
                run_sum = 0
                for i in range(1,n):
                    run_sum = run_sum + (i+1-1)*X[i]
                b1 = run_sum/(n*(n-1))
            
                run_sum = 0
                for i in range(2,n):
                    run_sum = run_sum + (i+1-1)*(i+1-2)*X[i]
                b2 = run_sum/(n*(n-1)*(n-2))
            
                run_sum = 0
                for i in range(3,n):
                    run_sum = run_sum + (i+1-1)*(i+1-2)*(i+1-3)*X[i]
                b3 = run_sum/(n*(n-1)*(n-2)*(n-3))
            
                run_sum = 0
                for i in range(4,n):
                    run_sum = run_sum + (i+1-1)*(i+1-2)*(i+1-3)*(i+1-4)*X[i]
                b4 = run_sum/(n*(n-1)*(n-2)*(n-3)*(n-4))
            
                l1 = b0
                l2 = 2*b1-b0
                l3 = 6*b2-6*b1+b0
                l4 = 20*b3-30*b2+12*b1-b0
                LSK = l3/l2
                LKU = l4/l2
            
                # now assign our newly constructed DF with this value in the correct position
                PRMS_month_errors_mean_df.loc[year, site] = l1
                PRMS_month_errors_Lscale_df.loc[year, site] = l2
                PRMS_month_errors_Lskew_df.loc[year, site] = LSK
                PRMS_month_errors_Lkurt_df.loc[year, site] = LKU
            
            else:
                # if the site doesn't have 80% of the data for a certain month, make the entry a np.nan
                PRMS_month_errors_mean_df.loc[year, site] = np.nan
                PRMS_month_errors_Lscale_df.loc[year, site] = np.nan
                PRMS_month_errors_Lskew_df.loc[year, site] = np.nan
                PRMS_month_errors_Lkurt_df.loc[year, site] = np.nan
   
    # drop rows that are in the training set
    PRMS_month_errors_mean_df = PRMS_month_errors_mean_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    PRMS_month_errors_Lscale_df = PRMS_month_errors_Lscale_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    PRMS_month_errors_Lskew_df = PRMS_month_errors_Lskew_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    PRMS_month_errors_Lkurt_df = PRMS_month_errors_Lkurt_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    
    return PRMS_month_errors_mean_df, PRMS_month_errors_Lscale_df, PRMS_month_errors_Lskew_df, PRMS_month_errors_Lkurt_df

########## PRMS SNOWMELT FULL DATASET ##########
def PRMS_create_monthly_snowmeltcumsum_dfs(month):
    monthly_snowmelt_df = pd.read_csv('monthly_cumulativesnowmelt.csv')

    # we only want to keep the sites at which we have errors as well
    PRMS_error_df = pd.read_csv('calibrated_error_between_PRMS_and_observed.csv')
    for site in monthly_snowmelt_df.columns:
        if site not in PRMS_error_df.columns:
            monthly_snowmelt_df.drop(site, axis=1, inplace=True)
        
    # get these columns in the correct order
    cols_in_order = monthly_snowmelt_df.columns[1:].sort_values()
    monthly_snowmelt_df = pd.concat([monthly_snowmelt_df['Date'], monthly_snowmelt_df[cols_in_order]], axis=1)

    # set up the DF
    monthly_snowmelt_df['Date'] = pd.to_datetime(monthly_snowmelt_df['Date'])
    monthly_snowmelt_df['Year'] = monthly_snowmelt_df['Date'].dt.year
    monthly_snowmelt_df['Month'] = monthly_snowmelt_df['Date'].dt.month

    # set up dummy DFs for all the statistics we want (for some reason for loops aren't working)
    # each DF just has years from 1916-2015 in the index, and each column name is a site number
    # these entries are all -99 for now, but they will eventually hold whatever L-moment is specified in the DF name
    
    
    #years = pd.Series(range(1916, 2016), name='Year')
    years = pd.Series(range(1951, 2016), name='Year')

    month_snowmelt_sum_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), 
                                                       columns=monthly_snowmelt_df.columns[1:-2])], axis=1)
    month_snowmelt_sum_df.set_index('Year', inplace=True)

    # choose the month of interest
    month_snowmelt_df = monthly_snowmelt_df[monthly_snowmelt_df['Month']==month]
    # get cumsum of snowmelt for each year
    for year in years:
        month_year_snowmelt_df = month_snowmelt_df[month_snowmelt_df['Year']==year]
        # get cumsum of snowmelt for each site
        for site in month_year_snowmelt_df.columns[1:-2]:
            # only find cumsum if the site has over 80% of available data in that given month
            if len(month_year_snowmelt_df[site].dropna()) >= 0.8*len(month_year_snowmelt_df):
                # now find snowmelt cumsum of this array
                X = np.array(month_year_snowmelt_df[site].dropna())
                cumsum = np.max(X)
            
                # now assign our newly constructed DF with this value in the correct position
                month_snowmelt_sum_df.loc[year, site] = cumsum
            else:
                # if the site doesn't have 80% of the data for a certain month, make the entry a np.nan
                month_snowmelt_sum_df.loc[year, site] = np.nan
    
    # drop rows that are in the training set
    month_snowmelt_sum_df = month_snowmelt_sum_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    return month_snowmelt_sum_df

########## PRMS RUNOFF FULL DATASET ##########
def PRMS_create_monthly_runoffcumsum_dfs(month):
    observed_flows_df = pd.read_csv('observed_flows.csv')

    # we only want to keep the sites at which we have errors as well
    PRMS_error_df = pd.read_csv('calibrated_error_between_PRMS_and_observed.csv')
    for site in observed_flows_df.columns:
        if site not in PRMS_error_df.columns:
            observed_flows_df.drop(site, axis=1, inplace=True)
        
    # get these columns in the correct order
    cols_in_order = observed_flows_df.columns[1:].sort_values()
    observed_flows_df = pd.concat([observed_flows_df['Date'], observed_flows_df[cols_in_order]], axis=1)

    # set up the DF
    observed_flows_df['Date'] = pd.to_datetime(observed_flows_df['Date'])
    observed_flows_df['Year'] = observed_flows_df['Date'].dt.year
    observed_flows_df['Month'] = observed_flows_df['Date'].dt.month

    # set up dummy DFs for all the statistics we want (for some reason for loops aren't working)
    # each DF just has years from 1915-2015 in the index, and each column name is a site number
    # these entries are all -99 for now, but they will eventually hold whatever L-moment is specified in the DF name
    
    #years = pd.Series(range(1915, 2016), name='Year')
    years = pd.Series(range(1951, 2016), name='Year')

    month_runoff_sum_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), 
                                                         columns=observed_flows_df.columns[1:-2])], axis=1)
    month_runoff_sum_df.set_index('Year', inplace=True)

    # choose the month of interest
    month_runoff_df = observed_flows_df[observed_flows_df['Month']==month]
    # get cumsum of snowmelt for each year
    for year in years:
        month_year_runoff_df = month_runoff_df[month_runoff_df['Year']==year]
        # get cumsum of snowmelt for each site
        for site in month_year_runoff_df.columns[1:-2]:
            # only find cumsum if the site has over 80% of available data in that given month
            if len(month_year_runoff_df[site].dropna()) >= 0.8*len(month_year_runoff_df):
                # now find runoff cumsum of this array
                X = np.array(month_year_runoff_df[site].dropna())
                cumsum = np.sum(X)
            
                # now assign our newly constructed DF with this value in the correct position
                month_runoff_sum_df.loc[year, site] = cumsum
            else:
                # if the site doesn't have 80% of the data for a certain month, make the entry a np.nan
                month_runoff_sum_df.loc[year, site] = np.nan
    
    # drop rows that are in the training set
    month_runoff_sum_df = month_runoff_sum_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    return month_runoff_sum_df

########## HBV L-MOMENTS FULL DATASET ##########
def HBV_create_monthly_Lmoment_dfs(error_type, month1, month2):
    # load in the correct DF based on the error type
    if error_type == 'differenced errors':
        HBV_errors_df = pd.read_csv('error_between_HBV_and_observed.csv')
    elif error_type == 'lambdas':
        HBV_errors_df = pd.read_csv('lambda_HBV.csv')
        
     # we only want to keep the sites at which we have errors as well
    PRMS_error_df = pd.read_csv('calibrated_error_between_PRMS_and_observed.csv')
    for site in HBV_errors_df.columns:
        if site not in PRMS_error_df.columns:
            HBV_errors_df.drop(site, axis=1, inplace=True)
        
    # get these columns in the correct order
    cols_in_order = HBV_errors_df.columns[1:].sort_values()
    HBV_errors_df = pd.concat([HBV_errors_df['Date'], HBV_errors_df[cols_in_order]], axis=1)
        
    # set up the error DF
    HBV_errors_df['Date'] = pd.to_datetime(HBV_errors_df['Date'])
    HBV_errors_df['Year'] = HBV_errors_df['Date'].dt.year
    HBV_errors_df['Month'] = HBV_errors_df['Date'].dt.month
    
    # set up dummy DFs for all the L-moments we want (for some reason for loops aren't working)
    # each DF just has years from 1951-2015 in the index, and each column name is a site number
    # these entries are all -99 for now, but they will eventually hold whatever L-moment is specified in the DF name
    years = pd.Series(range(1951, 2016), name='Year')

    HBV_month_errors_mean_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), columns=HBV_errors_df.columns[1:-2])],
                                          axis=1)
    HBV_month_errors_mean_df.set_index('Year', inplace=True)

    HBV_month_errors_Lscale_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), columns=HBV_errors_df.columns[1:-2])],
                                          axis=1)
    HBV_month_errors_Lscale_df.set_index('Year', inplace=True)

    HBV_month_errors_Lskew_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), columns=HBV_errors_df.columns[1:-2])],
                                          axis=1)
    HBV_month_errors_Lskew_df.set_index('Year', inplace=True)

    HBV_month_errors_Lkurt_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), columns=HBV_errors_df.columns[1:-2])],
                                          axis=1)
    HBV_month_errors_Lkurt_df.set_index('Year', inplace=True)
    
    # choose the month of interest
    HBV_month_errors_df = HBV_errors_df[(HBV_errors_df['Month']==month1) | (HBV_errors_df['Month']==month2)]
    # get L-moment values for each year
    for year in years:
        HBV_month_year_errors_df = HBV_month_errors_df[HBV_month_errors_df['Year']==year]
        # get L-moment values for each site
        for site in HBV_month_year_errors_df.columns[1:-2]:
            # only calculate L-moments if the site has over 80% of available data in that given month
            if len(HBV_month_year_errors_df[site].dropna()) >= 0.8*len(HBV_month_year_errors_df):
                # now calculate L-moments of this array
                X = np.array(HBV_month_year_errors_df[site].dropna())
                n = len(X)
                X = np.sort(X)
            
                b0 = np.mean(X)
            
                run_sum = 0
                for i in range(1,n):
                    run_sum = run_sum + (i+1-1)*X[i]
                b1 = run_sum/(n*(n-1))
            
                run_sum = 0
                for i in range(2,n):
                    run_sum = run_sum + (i+1-1)*(i+1-2)*X[i]
                b2 = run_sum/(n*(n-1)*(n-2))
            
                run_sum = 0
                for i in range(3,n):
                    run_sum = run_sum + (i+1-1)*(i+1-2)*(i+1-3)*X[i]
                b3 = run_sum/(n*(n-1)*(n-2)*(n-3))
            
                run_sum = 0
                for i in range(4,n):
                    run_sum = run_sum + (i+1-1)*(i+1-2)*(i+1-3)*(i+1-4)*X[i]
                b4 = run_sum/(n*(n-1)*(n-2)*(n-3)*(n-4))
            
                l1 = b0
                l2 = 2*b1-b0
                l3 = 6*b2-6*b1+b0
                l4 = 20*b3-30*b2+12*b1-b0
                LSK = l3/l2
                LKU = l4/l2
            
                # now assign our newly constructed DF with this value in the correct position
                HBV_month_errors_mean_df.loc[year, site] = l1
                HBV_month_errors_Lscale_df.loc[year, site] = l2
                HBV_month_errors_Lskew_df.loc[year, site] = LSK
                HBV_month_errors_Lkurt_df.loc[year, site] = LKU
            
            else:
                # if the site doesn't have 80% of the data for a certain month, make the entry a np.nan
                HBV_month_errors_mean_df.loc[year, site] = np.nan
                HBV_month_errors_Lscale_df.loc[year, site] = np.nan
                HBV_month_errors_Lskew_df.loc[year, site] = np.nan
                HBV_month_errors_Lkurt_df.loc[year, site] = np.nan
    
    # drop rows that are in the training set
    HBV_month_errors_mean_df = HBV_month_errors_mean_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    HBV_month_errors_Lscale_df = HBV_month_errors_Lscale_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    HBV_month_errors_Lskew_df = HBV_month_errors_Lskew_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    HBV_month_errors_Lkurt_df = HBV_month_errors_Lkurt_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    
    return HBV_month_errors_mean_df, HBV_month_errors_Lscale_df, HBV_month_errors_Lskew_df, HBV_month_errors_Lkurt_df

########## HBV SNOWMELT FULL DATASET ##########
def HBV_create_monthly_snowmeltcumsum_dfs(month):
    monthly_snowmelt_df = pd.read_csv('monthly_cumulativesnowmelt.csv')

    # we only want to keep the sites at which we have errors as well
    PRMS_error_df = pd.read_csv('calibrated_error_between_PRMS_and_observed.csv')
    for site in monthly_snowmelt_df.columns:
        if site not in PRMS_error_df.columns:
            monthly_snowmelt_df.drop(site, axis=1, inplace=True)
        
    # get these columns in the correct order
    cols_in_order = monthly_snowmelt_df.columns[1:].sort_values()
    monthly_snowmelt_df = pd.concat([monthly_snowmelt_df['Date'], monthly_snowmelt_df[cols_in_order]], axis=1)

    # set up the DF
    monthly_snowmelt_df['Date'] = pd.to_datetime(monthly_snowmelt_df['Date'])
    monthly_snowmelt_df['Year'] = monthly_snowmelt_df['Date'].dt.year
    monthly_snowmelt_df['Month'] = monthly_snowmelt_df['Date'].dt.month

    # set up dummy DFs for all the statistics we want (for some reason for loops aren't working)
    # each DF just has years from 1951-2015 in the index, and each column name is a site number
    # these entries are all -99 for now, but they will eventually hold whatever L-moment is specified in the DF name
    years = pd.Series(range(1951, 2016), name='Year')

    month_snowmelt_sum_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), 
                                                       columns=monthly_snowmelt_df.columns[1:-2])], axis=1)
    month_snowmelt_sum_df.set_index('Year', inplace=True)

    # choose the month of interest
    month_snowmelt_df = monthly_snowmelt_df[monthly_snowmelt_df['Month']==month]
    # get cumsum of snowmelt for each year
    for year in years:
        month_year_snowmelt_df = month_snowmelt_df[month_snowmelt_df['Year']==year]
        # get cumsum of snowmelt for each site
        for site in month_year_snowmelt_df.columns[1:-2]:
            # only find cumsum if the site has over 80% of available data in that given month
            if len(month_year_snowmelt_df[site].dropna()) >= 0.8*len(month_year_snowmelt_df):
                # now find snowmelt cumsum of this array
                X = np.array(month_year_snowmelt_df[site].dropna())
                cumsum = np.max(X)
            
                # now assign our newly constructed DF with this value in the correct position
                month_snowmelt_sum_df.loc[year, site] = cumsum
            else:
                # if the site doesn't have 80% of the data for a certain month, make the entry a np.nan
                month_snowmelt_sum_df.loc[year, site] = np.nan
    
    # drop rows that are in the training set
    month_snowmelt_sum_df = month_snowmelt_sum_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    
    return month_snowmelt_sum_df

########## HBV RUNOFF FULL DATASET ##########
def HBV_create_monthly_runoffcumsum_dfs(month):
    observed_flows_df = pd.read_csv('observed_flows.csv')

    # we only want to keep the sites at which we have errors as well
    PRMS_error_df = pd.read_csv('calibrated_error_between_PRMS_and_observed.csv')
    for site in observed_flows_df.columns:
        if site not in PRMS_error_df.columns:
            observed_flows_df.drop(site, axis=1, inplace=True)
        
    # get these columns in the correct order
    cols_in_order = observed_flows_df.columns[1:].sort_values()
    observed_flows_df = pd.concat([observed_flows_df['Date'], observed_flows_df[cols_in_order]], axis=1)

    # set up the DF
    observed_flows_df['Date'] = pd.to_datetime(observed_flows_df['Date'])
    observed_flows_df['Year'] = observed_flows_df['Date'].dt.year
    observed_flows_df['Month'] = observed_flows_df['Date'].dt.month

    # set up dummy DFs for all the statistics we want (for some reason for loops aren't working)
    # each DF just has years from 1951-2015 in the index, and each column name is a site number
    # these entries are all -99 for now, but they will eventually hold whatever L-moment is specified in the DF name
    years = pd.Series(range(1951, 2016), name='Year')

    month_runoff_sum_df = pd.concat([years, pd.DataFrame(-99, index=range(len(years)), 
                                                         columns=observed_flows_df.columns[1:-2])], axis=1)
    month_runoff_sum_df.set_index('Year', inplace=True)

    # choose the month of interest
    month_runoff_df = observed_flows_df[observed_flows_df['Month']==month]
    # get cumsum of snowmelt for each year
    for year in years:
        month_year_runoff_df = month_runoff_df[month_runoff_df['Year']==year]
        # get cumsum of snowmelt for each site
        for site in month_year_runoff_df.columns[1:-2]:
            # only find cumsum if the site has over 80% of available data in that given month
            if len(month_year_runoff_df[site].dropna()) >= 0.8*len(month_year_runoff_df):
                # now find runoff cumsum of this array
                X = np.array(month_year_runoff_df[site].dropna())
                cumsum = np.sum(X)
            
                # now assign our newly constructed DF with this value in the correct position
                month_runoff_sum_df.loc[year, site] = cumsum
            else:
                # if the site doesn't have 80% of the data for a certain month, make the entry a np.nan
                month_runoff_sum_df.loc[year, site] = np.nan
                
    # drop rows that are in the training set
    month_runoff_sum_df = month_runoff_sum_df.drop([1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005])
    
    return month_runoff_sum_df

########## MAKING THE FIRST STAT-HOLDING CSV FILE ##########
def making_csv_file(site):
    PRMS_Errors_csv = pd.DataFrame(np.nan, index=[0], columns=['Site Number','Site Name',
                                                               'Location','Years_in_Record','Drainage_Area_sqmi', 'Long',
                                                               'R2_Snow_Runoff',
                                                           'AMS_Per_MarApr','AMS_Upper50_Per_MarApr','VIF', 
                                                              'snow_lag_05', 'snow_lag_10', 'snow_lag_20',
                                                              'nosnow_lag_05', 'nosnow_lag_10', 'nosnow_lag_20',
                                                               'Mean_MR_AdjR2',
                                                           'Mean_MR_Snow_Coef','Mean_MR_Snow_Pval','Mean_MR_Runoff_Coef',
                                                           'Mean_MR_Runoff_Pval','Mean_SR_Snow_R2','Mean_SR_Snow_Coef',
                                                               'Mean_SR_Snow_Pval','Mean_SR_Runoff_R2',
                                                           'LSc_MR_AdjR2','LSc_MR_Snow_Coef','LSc_MR_Snow_Pval',
                                                           'LSc_MR_Runoff_Coef','LSc_MR_Runoff_Pval','LSc_SR_Snow_R2',
                                                               'LSc_SR_Snow_Coef','LSc_SR_Snow_Pval',
                                                           'LSc_SR_Runoff_R2','LSk_MR_AdjR2','LSk_MR_Snow_Coef',
                                                           'LSk_MR_Snow_Pval','LSk_MR_Runoff_Coef','LSk_MR_Runoff_Pval',
                                                           'LSk_SR_Snow_R2', 'LSk_SR_Snow_Coef', 'LSk_SR_Snow_Pval', 
                                                               'LSk_SR_Runoff_R2','LKur_MR_AdjR2',
                                                           'LKur_MR_Snow_Coef','LKur_MR_Snow_Pval','LKur_MR_Runoff_Coef',
                                                           'LKur_MR_Runoff_Pval','LKur_SR_Snow_R2','LKur_SR_Snow_Coef',
                                                               'LKur_SR_Snow_Pval','LKur_SR_Runoff_R2'])
    PRMS_Errors_csv['Site Number'] = site
    
    gages3_df = pd.read_csv('GagesIII_info.csv')
    # for some reason this column didn't transfer over as a string type
    gages3_df['GAGEID text'] = gages3_df['GAGEID text'].astype(str)

    #print(gages3_df[gages3_df['GAGEID text']==site])
    
    if gages3_df[gages3_df['GAGEID text']==site]['Description'].empty:
        PRMS_Errors_csv['Site Name'] = 'NOT IN GAGESIII_Info.csv!'
        PRMS_Errors_csv['Location'] = 'NOT IN GAGESIII_Info.csv!'
        PRMS_Errors_csv['Drainage_Area_sqmi'] = 'NOT IN GAGESIII_Info.csv!'
        PRMS_Errors_csv['Long'] = 'NOT IN GAGESIII_Info.csv!'
        
    else:
        PRMS_Errors_csv['Site Name'] = gages3_df[gages3_df['GAGEID text']==site]['Description'].values[0][29:-3].title() 
        PRMS_Errors_csv['Drainage_Area_sqmi'] = gages3_df[gages3_df['GAGEID text']==site]['Drainage Area (mi2)'].values[0]
        PRMS_Errors_csv['Long'] = gages3_df[gages3_df['GAGEID text']==site]['LONG'].values[0]
        
        clusters_coordinates = pd.read_csv('coordinates_and_clusters.csv')
        clusters_coordinates['GAGEID text'] = clusters_coordinates['GAGEID text'].astype(str)
        clusters_coordinates = clusters_coordinates.drop(columns=['LONG','LAT'])
        clusters_coordinates.loc[len(clusters_coordinates)] = ['1097480', 2]
        clusters_coordinates.loc[len(clusters_coordinates)] = ['1106500', 0]
        clusters_coordinates.loc[len(clusters_coordinates)] = ['1111050', 2]
        cluster_mapping = {0:'South Shore', 1:'Pioneer Valley', 2:'Central', 3:'Berkshires', 4:'Boston and North Shore'}
        clusters_coordinates['cluster'] = clusters_coordinates['cluster'].map(cluster_mapping)
        PRMS_Errors_csv['Location'] = clusters_coordinates[clusters_coordinates['GAGEID text']==site]['cluster'].values[0]
    
    return PRMS_Errors_csv

########## PPCC plot and % of annual maximums in Mar/Apr ##########
def ppcc_plot(site, csv_df):
    # finding annual max DF
    observed_flows_df = pd.read_csv('observed_flows.csv')
    observed_flows_df['Date'] = pd.to_datetime(observed_flows_df['Date'])
    observed_flows_df['Year'] = observed_flows_df['Date'].dt.year
    observed_flows_df['Month'] = observed_flows_df['Date'].dt.month
    annual_max_df = observed_flows_df[observed_flows_df['Date'] < '2016-01-01']
    annual_max_df.set_index('Date', inplace=True)
    annual_max_df = annual_max_df[[site,'Year']]
    annual_max_flow = annual_max_df[site].resample('A').max()
    annual_max_date = annual_max_df[site].resample('A').apply(lambda x: x.idxmax())
    annual_max_df = pd.DataFrame({'Flow':annual_max_flow, 'Date':annual_max_date, 'Year':annual_max_df['Year'].unique()}).reset_index(drop=True)
    # drop the years that have less than 80% of observations
    for year in annual_max_df['Year'].unique():
        if len(observed_flows_df[observed_flows_df['Year']==year][site].dropna()) < 0.8*len(observed_flows_df[observed_flows_df['Year']==year]):
            annual_max_df = annual_max_df[annual_max_df['Year'] != year]
    annual_max_df = annual_max_df.sort_values(by=['Flow']).dropna().reset_index(drop=True)
    annual_max_df['Rank'] = annual_max_df['Flow'].rank(method='max')
    annual_max_df['Non-exceedance probability'] = annual_max_df['Rank'] / (len(annual_max_df['Rank'])+1)
    annual_max_df['Month-Year'] = annual_max_df['Date'].dt.strftime('%m-%Y')
    annual_max_df['Month'] = annual_max_df['Date'].dt.month
    annual_max_df['Year'] = annual_max_df['Date'].dt.year
    annual_max_df['z-score'] = stats.norm.ppf(annual_max_df['Non-exceedance probability'])

    # plotting
    #MarApr_annual_max_df = annual_max_df[(annual_max_df['Month']==3) | (annual_max_df['Month']==4)]
    #other_annual_max_df = annual_max_df[(annual_max_df['Month']!=3) & (annual_max_df['Month']!=4)]
    #plt.figure(figsize=(10,8))
    #plt.scatter(MarApr_annual_max_df['z-score'], MarApr_annual_max_df['Flow'], color='orange', label='In March or April', 
    #            zorder=10, alpha=0.5)
    #plt.scatter(other_annual_max_df['z-score'], other_annual_max_df['Flow'], color='k', alpha=0.5)
    #plt.xlabel('Z-score', size=13)
    #plt.ylabel('Flow', size=13)
    #plt.title('Annual maximum PPCC plot at site '+site, size=15)
    #plt.legend(prop={'size':13})
    #plt.show()
#
    #plt.figure(figsize=(10,8))
    #plt.scatter(MarApr_annual_max_df['z-score'], MarApr_annual_max_df['Flow'], color='orange', label='In March or April', 
    #            zorder=10, alpha=0.5)
    #plt.scatter(other_annual_max_df['z-score'], other_annual_max_df['Flow'], color='k', alpha=0.5)
    #plt.yscale('log')
    #plt.xlabel('Z-score', size=13)
    #plt.ylabel('Flow', size=13)
    #plt.title('Annual maximum PPCC plot at site '+site, size=15)
    #plt.legend(prop={'size':13})
    #plt.show()
    
    AnnMax_MarApr = (len(annual_max_df[(annual_max_df['Month'] == 3) | (annual_max_df['Month'] == 4)])/len(annual_max_df))*100
    top_floods = annual_max_df.iloc[int(len(annual_max_df)/2):]
    top50_AnnMax_MarApr = (len(top_floods[(top_floods['Month'] == 3) | (top_floods['Month'] == 4)])/len(top_floods))*100
    
    #print('Percentage of annual maximums in March or April: '+str(round(AnnMax_MarApr,2))+'%')
    #print('Percentage of top 50% of annual maximums in March or April: '+str(round(top50_AnnMax_MarApr,2))+'%')
    
    csv_df['AMS_Per_MarApr'] = AnnMax_MarApr
    csv_df['AMS_Upper50_Per_MarApr'] = top50_AnnMax_MarApr
    
    return csv_df

########## NORMALIZING ARRAYS ##########
def normalize_array(df, site, model_name):
    if model_name == 'PRMS':
        # snip the array since not all of them start at the same time
        cut_df = df.loc[1916:]
    if model_name == 'HBV':
        cut_df = df.copy()
    site_array = cut_df[site].values
    
    # normalize this by subtracting the mean and dividing by the std
    mean = np.nanmean(site_array)
    std = np.nanstd(site_array)
    normalized_array = (site_array - mean)/std
    
    return normalized_array

########## UNIVARIATE LINEAR REGRESSION ##########
def linear_regression(x, y):
    # get rid of NaNs
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    # In order to have an intercept, we need to add a column of 1's to x
    x2 = sm.add_constant(x)

    # Fit the simple linear regression
    sm_model = sm.OLS(y, x2)
    results = sm_model.fit()

    return results

########## LINEAR REGRESSION SCATTERPLOT ##########
def scatterplot_Lmoment_climate(Lmoment_df, climate_df, Lmoment_name, climate_name, error_or_lambdas, month_name, site, model_name):
    fig = plt.figure(figsize=(10,8))
    
    if model_name == 'PRMS':
        # snip the array since not all of them start at the same time
        cut_Lmoment_df = Lmoment_df.loc[1916:]
        cut_climate_df = climate_df.loc[1916:]
    if model_name == 'HBV':
        cut_Lmoment_df = Lmoment_df.copy()
        cut_climate_df = climate_df.copy()

    plt.scatter(cut_climate_df[site], cut_Lmoment_df[site], color='gray')
        
    # get linear regression line
    normalized_Lmoment = normalize_array(Lmoment_df, site, model_name)
    normalized_climate = normalize_array(climate_df, site, model_name)
    
    res = linear_regression(x=normalized_climate, y=normalized_Lmoment)

    # find slope of line
    intercept, slope = res.params
    r2 = res.rsquared
    p_val = res.pvalues[1]
    
    # plot line
    if p_val <= 0.05:
        # get full linear regression line to plot
        full_x = cut_climate_df[site].values.flatten()
        full_y = cut_Lmoment_df[site].values.flatten()
        full_mask = ~np.isnan(full_x) & ~np.isnan(full_y)
        full_x = full_x[full_mask]
        full_y = full_y[full_mask]
        full_x_with_constant = sm.add_constant(full_x)
        full_model = sm.OLS(full_y, full_x_with_constant)
        full_results = full_model.fit()
        full_intercept, full_slope = full_results.params
        full_regression_line = full_intercept + full_slope*full_x
        plt.plot(full_x, full_regression_line, ls='--', color='r')
    
    plt.plot([],[],' ', label=f'R$^2$ = {r2:.2f}\n\u03B2 = {slope:.2f}\np-value = {p_val:.2f}')
    plt.legend(loc='best', fontsize=13, labelcolor='r')

    plt.ylabel(Lmoment_name, size=13)
    plt.xlabel(climate_name,size=13)

    plt.title(Lmoment_name+' of '+model_name+' '+error_or_lambdas+' vs '+climate_name+' in '+month_name+' at site '+site, size=15)
    

    
    return fig

########## MULTIVARIATE LINEAR REGRESSION ##########
def print_multilinear_regression(snowmelt_df, runoff_df, Lmoment_df, site, Lmoment_name, model_name):
    normalized_snowmelt = normalize_array(snowmelt_df, site, model_name)
    normalized_runoff = normalize_array(runoff_df, site, model_name)
    normalized_Lmoment = normalize_array(Lmoment_df, site, model_name)
    
    # get rid of NaNs
    mask = ~np.isnan(normalized_snowmelt) & ~np.isnan(normalized_runoff) & ~np.isnan(normalized_Lmoment)
    normalized_snowmelt = normalized_snowmelt[mask]
    normalized_runoff = normalized_runoff[mask]
    normalized_Lmoment = normalized_Lmoment[mask]
    
    X = pd.DataFrame({'Snowmelt':normalized_snowmelt, 'Runoff':normalized_runoff})
    y = normalized_Lmoment
    
    # In order to have an intercept, we need to add a column of 1's to X
    X2 = X.assign(Intercept=1)

    sm_model = sm.OLS(y, X2)
    results = sm_model.fit()
    
    print('Multi-feature model:')
    if results.params[1] >= 0:
        print(Lmoment_name+' = '+str(round(results.params[0],2))+'*Snowmelt+'+str(round(results.params[1],2))+'*Runoff')
    else:
        print(Lmoment_name+' = '+str(round(results.params[0],2))+'*Snowmelt-'+str(round(np.abs(results.params[1]),2))+'*Runoff')
    print((len(Lmoment_name)+8)*' '+'p = '+str(round(results.pvalues[0],2))+8*' '+'p = '+str(round(results.pvalues[1],2)))
    print(f'Adjusted R2 = {results.rsquared_adj:.2f}')
    print(f'R2 = {results.rsquared:.2f}')

########## PUTTING LINEAR REGRESSION STATS INTO OUR CSV FILE ##########
def put_stats_into_csv(csv_df, Lmoment_dfs, snowmelt_df, runoff_df, site, model_name):
    mean_df = Lmoment_dfs[0]
    Lscale_df = Lmoment_dfs[1]
    Lskew_df = Lmoment_dfs[2]
    Lkurt_df = Lmoment_dfs[3]
    
    # normalize these
    normalized_mean = normalize_array(mean_df, site, model_name)
    normalized_Lscale = normalize_array(Lscale_df, site, model_name)
    normalized_Lskew = normalize_array(Lskew_df, site, model_name)
    normalized_Lkurt = normalize_array(Lkurt_df, site, model_name)
    normalized_snowmelt = normalize_array(snowmelt_df, site, model_name)
    normalized_runoff = normalize_array(runoff_df, site, model_name)
    
    stats_array = np.array([])
    # run linear regressions
    for normalized_Lmoment in [normalized_mean, normalized_Lscale, normalized_Lskew, normalized_Lkurt]:
        # find univariate statistics
        res_snowmelt = linear_regression(x=normalized_snowmelt, y=normalized_Lmoment)
        r2_snowmelt = res_snowmelt.rsquared
        coef_snowmelt = res_snowmelt.params[1]
        pval_snowmelt = res_snowmelt.pvalues[1]
        
        res_runoff = linear_regression(x=normalized_runoff, y=normalized_Lmoment)
        r2_runoff = res_runoff.rsquared
        
        # find multivariate statistics
        # get rid of NaNs
        mask = ~np.isnan(normalized_snowmelt) & ~np.isnan(normalized_runoff) & ~np.isnan(normalized_Lmoment)
        normalized_snowmelt_mask = normalized_snowmelt[mask]
        normalized_runoff_mask = normalized_runoff[mask]
        normalized_Lmoment_mask = normalized_Lmoment[mask]
    
        X = pd.DataFrame({'Snowmelt':normalized_snowmelt_mask, 'Runoff':normalized_runoff_mask})
        y = normalized_Lmoment_mask
    
        # In order to have an intercept, we need to add a column of 1's to X
        X2 = X.assign(Intercept=1)

        sm_model = sm.OLS(y, X2)
        results = sm_model.fit()
        
        mv_slope_snowmelt = results.params[0]
        mv_slope_runoff = results.params[1]
        
        mv_pval_snowmelt = results.pvalues[0]
        mv_pval_runoff = results.pvalues[1]
        
        mv_adjr2 = results.rsquared_adj
        
        # put all these statistics into an array
        current_stats_array = np.array([mv_adjr2, mv_slope_snowmelt, mv_pval_snowmelt, mv_slope_runoff, mv_pval_runoff, 
                                       r2_snowmelt, coef_snowmelt, pval_snowmelt, r2_runoff])
        
        # append these to the overall array that we have
        stats_array = np.append(stats_array, current_stats_array)
    
    #put into csv
    csv_df.iloc[:, 16:] = stats_array
    
    return csv_df

########## VIF CALCULATION ##########
def calc_VIF(snowmelt_df, runoff_df, model_name, site):
    normalized_snowmelt = normalize_array(df=snowmelt_df, site=site, model_name=model_name)
    normalized_runoff = normalize_array(df=runoff_df, site=site, model_name=model_name)

    mask = ~np.isnan(normalized_snowmelt) & ~np.isnan(normalized_runoff)
    normalized_snowmelt = normalized_snowmelt[mask]
    normalized_runoff = normalized_runoff[mask]

    feature_df = pd.DataFrame({'Snowmelt':normalized_snowmelt, 'Runoff':normalized_runoff})

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = feature_df.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(feature_df.values, i) for i in range(len(feature_df.columns))]
    
    return vif_data

########## EXPORTING CSV FILES ##########
def export_csv_files(PRMS_Errors_csv, PRMS_Lambdas_csv, HBV_Errors_csv, HBV_Lambdas_csv, site):
    os.makedirs('../sites_HBV/'+site, exist_ok=True)

    PRMS_Errors_csv.to_excel('../sites_HBV/'+site+'/PRMS_Differences.xlsx', index=False)
    PRMS_Lambdas_csv.to_excel('../sites_HBV/'+site+'/PRMS_Lambdas.xlsx', index=False)
    HBV_Errors_csv.to_excel('../sites_HBV/'+site+'/HBV_Differences.xlsx', index=False)
    HBV_Lambdas_csv.to_excel('../sites_HBV/'+site+'/HBV_Lambdas.xlsx', index=False)
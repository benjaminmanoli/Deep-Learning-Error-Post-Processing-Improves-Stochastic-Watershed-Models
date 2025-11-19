import os
import pandas as pd
import numpy as np

from statsmodels.tsa.ar_model import AutoReg
from scipy import stats
from scipy.stats import norm

import CRPS.CRPS as pscore

model = 'HBV'
site = '1176000'
m = 100 # m is the number of random AR parameter sets, Beta, generated for considering AR model uncertainty.
n = 100 # n is the number of stochastic streamflow realizations generated for each Beta set
k = 350

################################################################### FUNCTIONS ##############################################################
#Function to determine water year
def water_year(date):
    if date.month >=10:
        return date.year+1
    else:
        return date.year

def calc_NSE(obs,sim):
    """ Returns Nash-Sutcliffe Efficiency
    Input:
    obs - array of observed streamflow record
    sim - array of simulated streamflow record """
    NSE = 1 - (np.sum((obs-sim)**2)/np.sum((obs-np.mean(obs))**2))
    bias_ratio = (np.mean(sim)-np.mean(obs))/np.std(obs)
    variance_ratio = np.std(sim)/np.std(obs)
    correlation = np.corrcoef(sim, obs)[0,1]
    return NSE, bias_ratio, variance_ratio, correlation

def wilson(p,realization):
    '''
    wilson hilforty equation for fitting P3 and finding the recurance interval floods

    Parameters
    ----------
    p : quantile

    Returns
    -------
    Xp : quantile flood

    '''
    zp=norm.ppf(p)
    Z=np.log10(realization)
    Kp = (2/Z.skew())*((1 + ((Z.skew()*zp)/6) - ((Z.skew()**2)/36))**3) - (2/Z.skew())
    Xp=10**(Z.mean() + Z.std()*Kp)
    return Xp

#################################################### LOAD IN DATA AND ADJUST TO DATES WE WANT ############################################
df = pd.read_csv('Deterministic_models/Quaboag_'+model+'.csv')
df['date'] = pd.to_datetime(df['date'])

########################################################### RUNNING SWM ON HISTORY ###################################################
Epsilon = np.load('Historical_SWM_data/'+model+'_Epsilon.npy')
Beta = np.load('Historical_SWM_data/'+model+'_Beta.npy')
OutlierDates_1955 = pd.date_range(start='1955-08-18', end='1955-08-31', freq='D').tolist()
OutlierDates = OutlierDates_1955
# give larger buffer for SWM
OutlierDates_1955_SWM_PreFlood = pd.date_range(end='1955-08-17', periods=7, freq='D').tolist()
OutlierDates_1955_SWM_PostFlood = pd.date_range(start='1955-09-01', periods=7, freq='D').tolist()
OutlierDates_SWM = OutlierDates + OutlierDates_1955_SWM_PreFlood + OutlierDates_1955_SWM_PostFlood
Tyr_flood_yrs = np.concatenate((np.arange(10, 101), np.array([125, 150, 175, 200, 250, 300, 350, 400, 450, 500])))

########################################################### RUNNING THE BOOTSTRAP ###################################################
YeArS_10 = pd.read_csv('Bootstrap_data/Years_10.csv')
# IM RIGHT ABOVER HERE##############################################################################################

####################################################### RUNNING SWM ON BOOTSTRAPPED DATA ###################################################
run_ID = int(os.environ['SLURM_ARRAY_TASK_ID'])
mag='10'
for mag_df in [YeArS_10]:
    for bootstrap in range(run_ID*20, run_ID*20+20):
        bootstrap_df = pd.DataFrame()
        date_holder = np.empty(0, dtype='datetime64[ns]')
        Qgage_holder = np.array([])
        Qmodel_holder = np.array([])
        for year in mag_df[str(bootstrap)]:
            dates = df[df['water_year']==year]['date'].values
            Qgages = df[df['water_year']==year]['Qgage'].values
            Qmodels = df[df['water_year']==year]['Qmodel'].values
            date_holder = np.concatenate([date_holder, dates])
            Qgage_holder = np.concatenate([Qgage_holder, Qgages])
            Qmodel_holder = np.concatenate([Qmodel_holder, Qmodels])
        bootstrap_df['date'] = date_holder
        bootstrap_df['Qgage'] = Qgage_holder
        bootstrap_df['Qmodel'] = Qmodel_holder
        bootstrap_df['diff'] = bootstrap_df['Qmodel'] - bootstrap_df['Qgage']
        bootstrap_df['lambda'] = np.log(bootstrap_df['Qmodel'] / bootstrap_df['Qgage'])
        
        data_bootstrap = bootstrap_df.copy()
        E = [] #initialize empty list to store errors
        for t in range(0,len(data_bootstrap)): # loop through all time periods
            E.append(Epsilon - [0, data_bootstrap.loc[t, 'Qmodel']])
        Er = np.zeros((len(data_bootstrap),k)) #initialize empty 2d array with dimensions time periods x k (# nearest neighbors)
        for t in range(0,len(data_bootstrap)): #loop through all time periods
            E[t][:,1] = np.abs(E[t][:,1]) #get absolute values of errors
            x = E[t][np.argsort(E[t][:, 1])] #sort errors by magnitude
            Er[t,:] = x[0:k,0] # get the AR residuals from the k closest flows
        R = np.zeros((len(data_bootstrap),n*m)) #initialize empty 2d array with dimensions time periods x # realizations
        for i in range(0,n*m): #loop through all realizations
            I = np.random.randint(low=0, high=k, size=len(data_bootstrap)) #randomly identify indices
            R[:,i] = Er[tuple(np.arange(0,len(data_bootstrap),1)),I] #sample from Er at random indices I
        E = R #set E to equal R
        # l is a matrix of [len(data),m*n] representing stochastic log-ratio errors         
        L = np.ones((len(data_bootstrap),m*n))*-99 #initialize empty array to populate below
        # for the first four time periods, we set values to be the deterministic model lambdas across all realizations
        # this is because we do not have enough time periods in the past to populate the AR3 model stochastically
        for i in range(0,m*n):
            L[0:3,i] = data_bootstrap['lambda'].iloc[0:3]
        for t in range(3,len(data_bootstrap)):
            if data_bootstrap['Qmodel'][t] <= 5:
                rule_point = -4.5
            if (data_bootstrap['Qmodel'][t] > 5) & (data_bootstrap['Qmodel'][t] <= 50):
                rule_point = -2.5
            if (data_bootstrap['Qmodel'][t] > 50) & (data_bootstrap['Qmodel'][t] <= 150):
                rule_point = -2
            if (data_bootstrap['Qmodel'][t] > 150) & (data_bootstrap['Qmodel'][t] <= 250):
                rule_point = -1.75
            if (data_bootstrap['Qmodel'][t] > 250) & (data_bootstrap['Qmodel'][t] <= 500):
                rule_point = -1.5
            if (data_bootstrap['Qmodel'][t] > 500) & (data_bootstrap['Qmodel'][t] <= 1000):
                rule_point = -1
            if (data_bootstrap['Qmodel'][t] > 1000) & (data_bootstrap['Qmodel'][t] <= 1500):
                rule_point = -0.75
            if data_bootstrap['Qmodel'][t] > 1500:
                rule_point = -0.5
            for i in range(0,m*n):
                B = Beta[i//m,:]
                L[t, i] = B[0] + \
                          B[1] * L[t-1, i] + \
                          B[2] * L[t-2, i] + \
                          B[3] * L[t-3, i] + \
                          E[t, i]
                if L[t, i] < rule_point:
                    reflection_diff = L[t, i] - rule_point
                    L[t, i] = rule_point - reflection_diff
                    E[t, i] = E[t, i] - (2*reflection_diff)
        # Q is a matrix of stochastically simulated streamflows of size [m*n, len(data)]
        # Just like lambda = log(Qmodel/Qgage), we now set Q = Qmodel/e^L where
        # L is stochastic lambdas and Q is stochastic streamflow
        Q = np.array(data_bootstrap['Qmodel']) / np.exp(L.T)
        
        SWM_Q = pd.DataFrame(Q.T)
        log_SWM_Q = np.log(SWM_Q)
        SWM_Q.insert(0, 'date', data_bootstrap['date'])
        log_SWM_Q.insert(0, 'date', data_bootstrap['date'])
        current_year_group = 0
        year_groups = []
        for row in range(len(SWM_Q)):
            current_row = SWM_Q.iloc[row, :]
            if current_row['date'].month != 9 or current_row['date'].day != 30:
                year_groups.append(current_year_group)
            else:
                year_groups.append(current_year_group)
                current_year_group += 1
        SWM_Q.insert(1, 'year groups', year_groups)
        log_SWM_Q.insert(1, 'year groups', year_groups)
        NoOutliers_SWM_Q = SWM_Q.set_index(SWM_Q['date']).drop(index=OutlierDates_SWM, errors='ignore').reset_index(drop=True)
        
        AnnualMax_df = NoOutliers_SWM_Q.groupby('year groups').max().iloc[:, 1:].reset_index(drop=True)

        AnnualMax_months_df = NoOutliers_SWM_Q.groupby('year groups').idxmax().map(lambda x: NoOutliers_SWM_Q['date'][x])\
                              .iloc[:, 1:].reset_index(drop=True)
        AnnualMax_months_df = AnnualMax_months_df.apply(lambda col: col.dt.month)

        SWM_Tyr_floods_df = pd.DataFrame(columns=AnnualMax_df.columns)
        for flood_T in Tyr_flood_yrs:
            Tyr_flood = np.zeros(len(AnnualMax_df.columns))
            for i in range(len(AnnualMax_df.columns)):
                Tyr_flood[i] = wilson(1-(1/flood_T), AnnualMax_df[i])
            SWM_Tyr_floods_df.loc[flood_T] = Tyr_flood
        
        rolling7 = SWM_Q.set_index(SWM_Q['date']).iloc[:, 2:].rolling(7).mean().reset_index(drop=True)
        rolling7.insert(0, 'year groups', SWM_Q['year groups'])   
        low7day_months_df = rolling7.groupby(rolling7['year groups']).idxmin().map(lambda x: SWM_Q['date'][x])\
                            .reset_index(drop=True)
        low7day_months_df = low7day_months_df.apply(lambda col: col.dt.month)
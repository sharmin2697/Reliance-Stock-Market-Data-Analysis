# IMPORT STATEMENTS
from FinalProjectFunctions import *
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import seaborn as sns
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
# ------------------------------------------------------------------------------------------

# SECTION 1 - DATA PRE-PROCESSING
# Load data
df = pd.read_csv('RELIANCE.csv', header=0)
# print(df.head())

# Data description
print('Data Description')
print(df.describe())
print(75*'-')
print(df.columns)
print(75*'-')
print(df.info())
print(75*'-')

# Copy data - for different parts of the project
data = df.copy()
data1 = df.copy()
data2 = df.copy()
data3 = df.copy()

# Format dates
new_date = []
for dat in df['Date']:
    d = dt.datetime.strptime(dat, "%m/%d/%Y")
    d = d.date()
    new_date.append(d.isoformat())
df.index = new_date
df = df.drop(['Date','Symbol','Series','Trades','Deliverable Volume','%Deliverble'], axis=1)

# Determine 'target' or 'dependent' variable
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date')
data = data.drop(['Symbol','Series','Trades', 'Deliverable Volume','%Deliverble'], axis=1)
dep_var = data['Close']

# Correlation matrix and Pearson's correlation coefficient
print('Correlation Matrix')
corr_matrix = df.corr()
print(corr_matrix)
print(75*'-')
corr_plot = sns.heatmap(corr_matrix,vmin=-1,vmax=1,center=0).set_title('Heatmap of the dataset')
plt.show()

# Plot of dependent variable vs time - original data
plt.figure()
plt.plot(dep_var, label='Closing Price of Stocks')
plt.xlabel('Year')
plt.ylabel('Price per Day')
plt.legend()
plt.title('Dependent Variable ("Close") vs Time')
plt.xticks(rotation=90)
plt.show()

# ACF calculation and plot of original data
lags = np.arange(1,21)
orig_acf = acf_values_df(dep_var,lags)
acf_plot(orig_acf,a='Dependent Variable')

# ADF-test 1 - original data
print('ADF-test on original dependent variable:')
adf_cal(dep_var)
print(75*'-')

# First Order Differencing
y_diff = dep_var.diff().dropna(axis=0)
y_diff.reset_index()

# ADF-test 2 - differenced data
print('ADF-test on differenced dependent variable:')
adf_cal(y_diff)
print(75*'-')

# Plot of dependent variable vs time - after differencing
plt.figure()
plt.plot(y_diff, label='Differenced Dependent Variable')
plt.xlabel('Year')
plt.ylabel('Magnitude')
plt.title('Dependent Variable vs Time - After First Order Differencing')
plt.xticks(rotation=90)
plt.legend()
plt.show()

# ACF calculation and plot of the differenced data
y_acf = acf_values_df(y_diff,lags)
acf_plot(y_acf,a='Dependent Variable')

# Data splitting into 80% - train set and 20% - test set
data = data.drop(columns='Close')
data = data.join(y_diff).set_index(data.index)
data = data.dropna(axis=0)
y = y_diff.to_frame()
y = y.reset_index()  # resets index and sets 'Date' as a column
train, test = train_test_split(y, shuffle=False, test_size=0.2)
# ------------------------------------------------------------------------------------------

# SECTION 2 - TIME SERIES DECOMPOSITION

Close = data1['Close'].to_frame()  # to_frame() converts this series to a dataframe

# Apply STL decomposition and plot
stl = STL(Close, period=52)
result = stl.fit()
fig = result.plot()
plt.show()

T = result.trend
S = result.seasonal
R = result.resid

# Plot
plt.figure()
plt.plot(T, label = 'Trend')
plt.plot(S, label = 'Seasonal')
plt.plot(R, label = 'Remainder')
plt.legend()
plt.title('Trend, Seasonality and Remainder')
plt.xlabel('Years')
plt.ylabel('Values')
plt.show()


# FUNCTION TO CALCULATE STRENGTH
def strength_stats(trend, seasonal, resid):
    ft = np.maximum(0,1 - (np.var(np.array(resid))/(np.var(np.array(trend+resid)))))
    fs = np.maximum(0,1 - (np.var(np.array(resid))/(np.var(np.array(seasonal+resid)))))
    return ft, fs

# CALCULATING STRENGTH OF TREND AND SEASONALITY
F_T_mul, F_S_mul = strength_stats(T,S,R)
print('The strength of Trend for data is:', F_T_mul)
print('The strength of Seasonality for data is:', F_S_mul)
print(75*'-')
# ------------------------------------------------------------------------------------------

# SECTION 3 - BASE MODELS
# Load data and format dates
data1['Date'] = pd.to_datetime(data1['Date'])

# Non-differenced data
new_data = data1[['Date', 'Close']]
train_nd, test_nd = train_test_split(new_data, shuffle=False, test_size=0.2)
# Differenced data - Already set above as train and test

# AVERAGE METHOD
avg_p, avg_f, res_avg, err_avg, mse_p_avg, mse_f_avg, var_p_avg, var_f_avg, \
res_acf_avg, Q_avg = call_avg(train, test, lags, a='Close', b='Date', c='Average')
avg_corr = correlation_coefficient_cal1(err_avg, test['Close'])

# NAIVE METHOD
nai_p, nai_f, res_nai, err_nai, mse_p_nai, mse_f_nai, var_p_nai, var_f_nai, \
res_acf_nai, Q_nai = call_naive(train, test, lags, a='Close', b='Date', c='Naive')
nai_corr = correlation_coefficient_cal1(err_nai, test['Close'])

# DRIFT METHOD
dri_p, dri_f, res_dri, err_dri, mse_p_dri, mse_f_dri, var_p_dri, var_f_dri, \
res_acf_dri, Q_dri = call_drift(train, test, lags, a='Close', b='Date', c='Drift')
dri_corr = correlation_coefficient_cal1(err_dri, test['Close'])

# SES METHOD
ses_p, ses_f, res_ses, err_ses, mse_p_ses, mse_f_ses, var_p_ses, var_f_ses, \
res_acf_ses, Q_ses = call_ses(train, test, 0.5, lags, a='Close', b='Date', c='SES')
ses_corr = correlation_coefficient_cal1(err_ses, test['Close'])
# ------------------------------------------------------------------------------------------

# HOLT-WINTER'S METHOD
# HOLT'S SEASONAL METHOD
hs_p, hs_f, res_hs, err_hs, mse_p_hs, mse_f_hs, var_p_hs, var_f_hs, \
res_acf_hs, Q_hs = call_holt_s(train,test,lags,a='Close',b='Date',c="Holt's Seasonal")
hs_corr = correlation_coefficient_cal1(err_hs, test['Close'])

# HOLT'S LINEAR METHOD - Non-Differenced Data
hl_p, hl_f, res_hl, err_hl, mse_p_hl, mse_f_hl, var_p_hl, var_f_hl, \
res_acf_hl, Q_hl = call_holt_l(train_nd,test_nd,lags,a='Close',b='Date',c="Holt's Linear")
hl_corr = correlation_coefficient_cal1(err_hl, test_nd['Close'])

# HOLT'S LINEAR METHOD - Differenced Data
hl_p_d, hl_f_d, res_hl_d, err_hl_d, mse_p_hl_d, mse_f_hl_d, var_p_hl_d, var_f_hl_d, \
res_acf_hl_d, Q_hl_d = call_holt_l(train,test,lags,a='Close',b='Date',c="Holt's Linear")
hl_corr_d = correlation_coefficient_cal1(err_hl_d, test['Close'])
# ------------------------------------------------------------------------------------------

# MULTIPLE LINEAR REGRESSION
# Load data
data2 = data2.drop(['Date','Symbol','Series','Trades', 'Deliverable Volume','%Deliverble'], axis=1)

# Select target
target = 'Close'
y = data2[target]
X = data2.drop(columns=target)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Create correlation matrix to find important features
# Update X_train and X_test with important features
corr_mat = df.corr()
features = corr_mat.index
new_features = features.drop(target,1)
X_train = X_train[new_features]
X_test = X_test[new_features]
# print(X_train.head(), X_train.columns)  # (4147, 8)
# print(X_test.head(), X_test.columns)  # (1037, 8)

# Create arrays using the split sets
y_trainarr = np.array(y_train).reshape(len(y_train),1)
y_testarr = np.array(y_test).reshape(len(y_test),1)
X_trainarr = X_train.to_numpy()
X_testarr = X_test.to_numpy()

# Add constant and begin modeling
X = sm.add_constant(X_train)
model1 = sm.OLS(y_train,X).fit()
# print(model1.summary())

# Feature Selection - Backward Stepwise Regression
print('BACKWARD STEPWISE REGRESSION')
# Drop Volume - based on the summary of model 1 it was seen that the p val of Volume is 0.769 > 0.01 or 0.05
model2, X_train2 = backward_reg(X_train,y_train,drop_feature='Volume')
# Drop Low - based on model 2 summary - the p valy (0.224) > 0.01 or 0.05
model3,X_train3 = backward_reg(X_train2,y_train,drop_feature='Low')
# Drop Turnover - p-val is 0.019 based on the model 3 summary
model4, X_train4 = backward_reg(X_train3,y_train,drop_feature='Turnover')

# Table with information
pd.set_option('display.max_rows', None,'display.max_columns', None)
stat_table = pd.DataFrame({'AIC': [model1.aic,model2.aic,model3.aic,model4.aic],
                           'BIC': [model1.bic,model2.bic,model3.bic,model4.bic],
                           'Adj. R-squared': [model1.rsquared_adj,model2.rsquared_adj,model3.rsquared_adj,
                                              model4.rsquared_adj]},
                          index=['Model 1','Model 2','Model 3','Model 4'])
print('The statistics of the models are as follows:')
print(stat_table)
print(75*'-')

# Statistics of final model 3
print('The statistics of Model 3 are as follows:')
print(model3.summary())
print('The F-test values are:')
print('F-value:', model3.fvalue,'F_p-value:', model3.f_pvalue)
print(75*'-')

# Modeling
fitted_val = model3.fittedvalues  # to help calculate residuals
# print('The Coefficients of Regression are:')
# print(model3.params)
x_test_new = X_test.drop(['Low','Volume'], axis=1)
X_1 = sm.add_constant(x_test_new)
prediction = model3.predict(X_1)

# Plot for 1-step prediction
plt.figure()
plt.plot(y_train, label='Train')
# plt.plot(y_test, label='Test')
plt.plot(prediction, label='Prediction')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Plot for Linear Regression')
plt.show()

# Model statistics
pred_err_lr = residual(y_train,fitted_val)
mse_pred_lr = mse_mlr(pred_err_lr)
acf_val_predn_lr = acf_values(pred_err_lr,lags)
acf_plot(acf_val_predn_lr,a='Linear Regression (Residual ACF)')
pred_var_lr = var_linreg(pred_err_lr,len(new_features))
Q_back = Q_val(y_trainarr, acf_val_predn_lr)

# Feature Selection - Forward Stepwise Regression
# Create a list of all the columns
feature_list = []
for name in df[new_features].columns:
    feature_list.append(name)

print('FORWARD STEPWISE REGRESSION')
X_trainf = pd.DataFrame()
# add Prev Close
modelf1, X_trainf1 = forward_reg(y_train,X_train,X_trainf,feature_list[0],name='Prev Close')
# add Open
modelf2, X_trainf2 = forward_reg(y_train,X_train,X_trainf1,feature_list[1],name='Open')
# add High
modelf3, X_trainf3 = forward_reg(y_train,X_train,X_trainf2,feature_list[2],name='High')
# add Low - this model was not taken into account because it gave p-values > 0.05
# modelf4, X_trainf4 = forward_reg(y_train,X_train,X_trainf3,feature_list[3],name='Low')
# print(modelf4.summary())
# add Last
modelf5, X_trainf5 = forward_reg(y_train,X_train,X_trainf3,feature_list[4],name='Last')
# add VWAP
modelf6, X_trainf6 = forward_reg(y_train,X_train,X_trainf5,feature_list[5],name='VWAP')
# add Volume
modelf7, X_trainf7 = forward_reg(y_train,X_train,X_trainf6,feature_list[6],name='Volume')
# add Turnover
modelf8, X_trainf8 = forward_reg(y_train,X_train,X_trainf7,feature_list[7],name='Turnover')

# Table with information
pd.set_option('display.max_rows', None,'display.max_columns', None)
stat_table = pd.DataFrame({'AIC': [modelf1.aic,modelf2.aic,modelf3.aic,modelf5.aic,modelf6.aic,modelf7.aic,modelf8.aic],
                           'BIC': [modelf1.bic,modelf2.bic,modelf3.bic,modelf5.bic,modelf6.bic,modelf7.bic,modelf8.bic],
                           'Adj. R-squared': [modelf1.rsquared_adj,modelf2.rsquared_adj,modelf3.rsquared_adj,modelf5.rsquared_adj,modelf6.rsquared_adj,modelf7.rsquared_adj,modelf8.rsquared_adj]},
                          index=['Model 1','Model 2','Model 3','Model 5','Model 6','Model 7','Model 8'])
print('The statistics of the models are as follows:')
print(stat_table)
print(75*'-')

# Statistics of final model 6
print('The statistics of Model 6 are as follows:')
print(modelf6.summary())
print('The F-test values are:')
print('F-value:', modelf6.fvalue,'F_p-value:', modelf6.f_pvalue)
print(75*'-')

# Modeling
fitted_val1 = modelf6.fittedvalues  # to help calculate residuals
x_test_new1 = X_test.drop(['Low','Volume','Turnover'], axis=1)
X_1 = sm.add_constant(x_test_new1)
prediction1 = modelf6.predict(X_1)

# Plot for 1-step prediction
plt.figure()
plt.plot(y_train, label='Train')
plt.plot(prediction1, label='Prediction')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Price')
plt.title('Plot for Linear Regression')
plt.show()

# Model statistics
pred_err1_lr = residual(y_train,fitted_val1)
mse_pred1_lr = mse_mlr(pred_err1_lr)
acf_val_predn1_lr = acf_values(pred_err1_lr,lags)
acf_plot(acf_val_predn1_lr,a='Linear Regression (Residual ACF)')
pred_var1 = var_linreg(pred_err1_lr,len(new_features))
Q_forw = Q_val(y_trainarr, acf_val_predn1_lr)
# ------------------------------------------------------------------------------------------

# ARMA() PROCESS - INCLUDING GPAC AND LM ALGORITHM
# Load data
data3['Date'] = pd.to_datetime(data3['Date'])

# Pre-process data
data3 = data3.set_index('Date')
data3 = data3.drop(['Symbol','Series','Trades', 'Deliverable Volume','%Deliverble'], axis=1)
dep_var = data3['Close']
y_diff = dep_var.diff().dropna(axis=0)
y_diff.reset_index()

data3 = data3.drop(columns='Close')
data3 = data3.join(y_diff).set_index(data3.index)
data3 = data3.dropna(axis=0)
train, test = train_test_split(y_diff, shuffle=False, test_size=0.2)

# ACF calculation
y_acf = acf_values_df(train,lags)

# GPAC
acf_list = list(sm.tsa.stattools.acf(y_acf))
gpac_cal(acf_list,8,8)

# LM Algorithm
def step_0(na,nb):
    theta = np.zeros(shape=(na+nb,1))
    return theta.flatten()

def white_noise_simulation(theta,na,y):
    num = [1] + list(theta[na:])
    den = [1] + list(theta[:na])
    while len(num) < len(den):
        num.append(0)
    while len(num) > len(den):
        den.append(0)
    system = (den, num, 1)
    tout, e = signal.dlsim(system, y)
    e = [a[0] for a in e]
    return np.array(e)

def step_1(theta,na,nb,delta,y):
    e = white_noise_simulation(theta,na,y)
    SSE = np.matmul(e.T, e)
    X_all = []
    for i in range(na+nb):
        theta_dummy = theta.copy()
        theta_dummy[i] = theta[i] + delta
        e_n = white_noise_simulation(theta_dummy,na,y)
        X_i = (e - e_n)/delta
        X_all.append(X_i)

    X = np.column_stack(X_all)
    A = np.matmul(X.T,X)
    g = np.matmul(X.T,e)
    return A,g,SSE

def step_2(A,mu,g,theta,na,y):
    I = np.identity(g.shape[0])
    theta_d = np.matmul(np.linalg.inv(A+(mu*I)),g)
    theta_new = theta + theta_d
    e_new = white_noise_simulation(theta_new,na,y)
    SSE_new = np.matmul(e_new.T,e_new)
    if np.isnan(SSE_new):
        SSE_new = 10 ** 10
    return SSE_new, theta_d, theta_new


with np.errstate(divide='ignore'):
    np.float64(1.0) / 0.0

def step_3(max_iterations, mu_max, na, nb, y, mu, delta):
    iteration_num = 0
    SSE = []
    theta = step_0(na, nb)
    while iteration_num < max_iterations:
        print('Iteration ', iteration_num)

        A, g, SSE_old = step_1(theta, na, nb, delta, y)
        print('old SSE : ', SSE_old)
        if iteration_num == 0:
            SSE.append(SSE_old)
        SSE_new, theta_d, theta_new = step_2(A, mu, g, theta, na, y)
        print('new SSE : ', SSE_new)
        SSE.append(SSE_new)

        if SSE_new < SSE_old:
            print('Norm of delta_theta :', np.linalg.norm(theta_d))
            if np.linalg.norm(theta_d) < 1e-3:
                theta_hat = theta_new
                e_var = SSE_new / (len(y) - A.shape[0])
                cov = e_var * np.linalg.inv(A)
                print('\n **** Algorithm Converged **** \n')
                return SSE, theta_hat, cov, e_var
            else:
                theta = theta_new
                mu = mu / 10

        while SSE_new >= SSE_old:
            mu = mu * 10
            if mu > mu_max:
                print('mu exceeded the max limit')
                return None, None, None, None
            SSE_new, theta_d, theta_new = step_2(A, mu, g, theta, na, y)

        theta = theta_new

        iteration_num+=1
        if iteration_num > max_iterations:
            print('Max iterations reached')
            return None, None, None, None


def SSEplot(SSE):
    plt.figure()
    plt.plot(SSE, label = 'Sum Squared Error')
    plt.xlabel('# of Iterations')
    plt.ylabel('Sum Squared Error')
    plt.legend()
    plt.show()


np.random.seed(10)
mu_factor = 10
delta = 1e-6
epsilon = 0.001
mu = 0.01
max_iterations = 100
mu_max = 1e10

# ARMA(2,2)
# Estimating parameters using LM algorithm
na = 2
nb = 2
SSE, est_params, cov, e_var = step_3(max_iterations, mu_max, na, nb, train, mu, delta)
print('Estimated parameters : ', est_params)
print('Estimated Covariance matrix : ', cov)
print('Estimated variance of error : ', e_var)

# SSE Plot
SSEplot(SSE)
confidence_interval(cov, na, nb, est_params)
zeros, poles = zeros_and_poles(est_params, na, nb)

# 1-step ahead prediction
y_hat_t_1 = []
for i in range(0,len(train)):
    if i==0:
        y_hat_t_1.append(-train[i]*est_params[0] + est_params[1]* train[i])
    elif i==1:
        y_hat_t_1.append(-train[i]*est_params[0] + est_params[1]*(train[i] - y_hat_t_1[i - 1] ) + est_params[2]*(train[i - 1]))
    else:
        y_hat_t_1.append( -train[i]*est_params[0] + est_params[1]*(train[i] - y_hat_t_1[i - 1] ) + est_params[2]*(train[i - 1] - y_hat_t_1[i-2]))

train_plot = train.to_frame()
predn = dataframe_create_arma(y_hat_t_1,train_plot,a='Close')
plt.figure()
plt.plot(train, label='Train')
plt.plot(predn, label='1-step prediction')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Value')
plt.title('Plot for ARMA(2,2) process')
plt.show()

# h-step ahead prediction
y_hat_t_h = []
for h in range(0,len(test)):
    if h==0:
        y_hat_t_h.append(-train[-1]*est_params[0] + est_params[1]*(train[-1] - y_hat_t_1[-2]) + est_params[2]*(train[-2]-y_hat_t_1[-3]))
    elif h==1:
         y_hat_t_h.append(-y_hat_t_h[h-1]*est_params[0] + est_params[1]*(y_hat_t_h[-1] - y_hat_t_1[-1]))
    else:
        y_hat_t_h.append(-y_hat_t_h[h-1]*est_params[0])

test_plot = test.to_frame()
forec = dataframe_create_arma(y_hat_t_h,test_plot,a='Close')
plt.figure()
plt.plot(test, label='Test')
plt.plot(forec, label='h-step prediction')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Value')
plt.xticks(rotation='vertical')
plt.title('Plot for ARMA(2,2) process')
plt.show()

# Model statistics
pred_err = residual_arma(train, y_hat_t_1)
forec_err = forecast_err_arma(test, y_hat_t_h)
mse_pred, mse_forec = mse(pred_err, forec_err)
var_pred, var_forec = var(pred_err, forec_err)
residual_acf = acf_values(pred_err, lags)
Q_val1 = Q_val(train, residual_acf)
acf_plot(residual_acf, a='ARMA(2,2) Process')
chi_square_test(Q_val1, 20, na, nb)

# ARMA(1,0)
# Estimating parameters using LM algorithm
na1 = 1
nb1 = 0
SSE1, est_params1, cov1, e_var1 = step_3(max_iterations, mu_max, na1, nb1, train, mu, delta)
print('Estimated parameters : ', est_params1)
print('Estimated Covariance matrix : ', cov1)
print('Estimated variance of error : ', e_var1)

# SSE Plot
SSEplot(SSE1)
confidence_interval(cov1, na1, nb1, est_params1)
zeros1, poles1 = zeros_and_poles(est_params1, na1, nb1)

# 1-step ahead prediction
y_hat_t_11 = []
for i in range(len(train)):
    y_hat_t_11.append(-est_params1[0] * train[i])

predn1 = dataframe_create_arma(y_hat_t_1,train_plot,a='Close')
plt.figure()
plt.plot(train, label='Train')
plt.plot(predn1, label='1-step prediction')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Value')
plt.title('Plot for ARMA(1,0) process')
plt.show()

# h-step ahead prediction
y_hat_t_h1 = []
for h in range(len(test)):
    if h == 0:
        y_hat_t_h1.append(-est_params1[0] * test[-1])
    else:
        y_hat_t_h1.append(-est_params1[0] * y_hat_t_h1[h-1])

forec1 = dataframe_create_arma(y_hat_t_h1,test_plot,a='Close')
plt.figure()
plt.plot(test, label='Test')
plt.plot(forec1, label='h-step prediction')
plt.legend()
plt.xlabel('Years')
plt.ylabel('Value')
plt.xticks(rotation='vertical')
plt.title('Plot for ARMA(1,0) process')
plt.show()

# Model statistics
pred_err1 = residual_arma(train, y_hat_t_11)
forec_err1 = forecast_err_arma(test, y_hat_t_h1)
mse_pred1, mse_forec1 = mse(pred_err1, forec_err1)
var_pred1, var_forec1 = var(pred_err1, forec_err1)
residual_acf1 = acf_values(pred_err1, lags)
Q_val2 = Q_val(train, residual_acf1)
acf_plot(residual_acf1, a='ARMA(1,0) Process')
chi_square_test(Q_val2, 20, na1, nb1)
# ------------------------------------------------------------------------------------------

# ARIMA MODELS
# ARIMA(2,1,2)
arima_fit1 = sm.tsa.arima.ARIMA(train_nd['Close'],order=(2,1,2)).fit()
arima_pred = arima_fit1.fittedvalues

plt.figure()
plt.plot(train_nd['Close'], label='Train')
plt.plot(arima_pred, label='1-step prediction')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.title('Plot for ARIMA(2,1,2)')
plt.show()

arima_forec = arima_fit1.forecast(len(test_nd['Close']))

plt.figure()
plt.plot(test_nd['Close'], label='Test')
plt.plot(arima_forec, label='h-step prediction')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.title('Plot for ARIMA(2,1,2)')
plt.show()

arima_res = residual(train_nd['Close'], arima_pred)
arima_forec_err = forecast_err(test_nd['Close'], arima_forec)
mse_pred_arima, mse_forec_arima = mse(arima_res, arima_forec_err)
var_pred_arima, var_forec_arima = var(arima_res, arima_forec_err)
residual_acf_arima = acf_values(arima_res, lags)
Q_val_arima = Q_val(train_nd['Close'], residual_acf_arima)
acf_plot(residual_acf_arima, a='ARIMA(2,1,2)')

# ARIMA(1,1,0)
arima_fit2 = sm.tsa.arima.ARIMA(train_nd['Close'],order=(1,1,0)).fit()
arima_pred1 = arima_fit2.fittedvalues

plt.figure()
plt.plot(train_nd['Close'], label='Train')
plt.plot(arima_pred1, label='1-step prediction')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.title('Plot for ARIMA(1,1,0)')
plt.show()

arima_forec1 = arima_fit2.forecast(len(test_nd['Close']))

plt.figure()
plt.plot(test_nd['Close'], label='Test')
plt.plot(arima_forec1, label='h-step prediction')
plt.xlabel('Index')
plt.ylabel('Price')
plt.legend()
plt.title('Plot for ARIMA(1,1,0)')
plt.show()

arima_res1 = residual(train_nd['Close'], arima_pred1)
arima_forec_err1 = forecast_err(test_nd['Close'], arima_forec1)
mse_pred_arima1, mse_forec_arima1 = mse(arima_res1, arima_forec_err1)
var_pred_arima1, var_forec_arima1 = var(arima_res1, arima_forec_err1)
residual_acf_arima1 = acf_values(arima_res1, lags)
Q_val_arima1 = Q_val(train_nd['Close'], residual_acf_arima1)
acf_plot(residual_acf_arima1, a='ARIMA(1,1,0)')
# ------------------------------------------------------------------------------------------

# FINAL MODEL SELECTION
print('STATISTICS')
pd.set_option('display.max_rows', None,'display.max_columns', None)
stat_table = pd.DataFrame({'MSE Prediction': [mse_p_avg,mse_p_nai,mse_p_dri,mse_p_ses,mse_p_hs,mse_p_hl,mse_p_hl_d,
                                              mse_pred_lr,mse_pred1_lr,mse_pred,mse_pred1,mse_pred_arima,mse_pred_arima1],
                           'MSE Forecast': [mse_f_avg,mse_f_nai,mse_f_dri,mse_f_ses,mse_f_hs,mse_f_hl,mse_f_hl_d,
                                            'N/A','N/A',mse_forec,mse_forec1,mse_forec_arima,mse_forec_arima1],
                           'Q Values': [Q_avg,Q_nai,Q_dri,Q_ses,Q_hs,Q_hl,Q_hl_d,Q_back,Q_forw,Q_val1,Q_val2,Q_val_arima,
                                        Q_val_arima1],
                           'Variance Residuals': [var_p_avg,var_p_nai,var_p_dri,var_p_ses,var_p_hs,var_p_hl,var_p_hl_d,
                                                  pred_var_lr,pred_var1,var_pred,var_pred1,var_pred_arima,var_pred_arima1],
                           'Variance Forecast Error': [var_f_avg,var_f_nai,var_f_dri,var_f_ses,var_f_hs,var_f_hl,
                                                       var_f_hl_d,'N/A','N/A',var_forec,var_forec1,var_forec_arima,var_forec_arima1]},
                          index=["Average","Naive","Drift","SES","Holt's Seasonal","Holt's Linear (Diff)","Holt's Linear (Non-Diff)",
                                 "Backward LR","Forward LR","ARMA(2,2)","ARMA(1,0)","ARIMA(2,1,2)","ARIMA(1,1,0)"])
print(stat_table)
# ------------------------------------------------------------------------------------------





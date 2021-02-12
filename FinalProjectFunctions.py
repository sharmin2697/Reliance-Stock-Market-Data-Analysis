import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.holtwinters as ets
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
from scipy.stats import chi2
from scipy import signal


# FUNCTION FOR ADF TEST
def adf_cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f "% result[0])
    print("p-value: %f" % result[1])
    print("Critical Values: ")
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    return None


# AUTOCORRELATION FUNCTION
def cal_acf_df(x,tau):
    x_bar = np.mean(x)
    den = 0
    num = 0
    for ele in range(len(x)):
        d = (x.iloc[ele] - x_bar)**2
        den += d
    for ele in range(tau,len(x)):
        n = (x.iloc[ele]-x_bar)*(x.iloc[ele-tau]-x_bar)
        num += n
    tau_n = num/den
    return tau_n


def acf_values_df(x, lag):
    tau_list = []
    for tau in range(len(lag)):
        tau_list.append(cal_acf_df(x,tau))
    # tau_list.pop(0)
    new_tau = []
    # print(tau_list)
    for each in tau_list:
        new_val = each**2
        new_tau.append(new_val)
    # print('The values for tau are as follows:')
    return new_tau


# ACF OF RESIDUALS FOR PLOTTING
def acf_plot(tau_list, a=''):
    sym_tau = tau_list[::-1]
    sym_tau.pop(-1)
    sym_tau.extend(tau_list)
    # print(sym_tau)
    plt.figure()
    plt.stem(sym_tau)
    plt.title('Auto-correlation Function of {}'.format(a))
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    return plt.show()


# FUNCTION TO CREATE DATAFRAME
def dataframe_create(prediction, test_set, a='', b=''):
    forecast1 = pd.DataFrame({a: test_set[a], b: prediction})
    return forecast1


# AVERAGE METHOD
def avg_method(train_s, test_s):
    T = len(train_s)
    test_len = len(test_s)
    # x = train_s.mean()
    # print(x)
    prediction = []
    sum = 0
    # prediction.append(train_s.cumsum())
    for i in range(T):
        sum += train_s.iloc[i]
        prediction.append(sum)
    last_val = prediction[-1]
    avg = last_val/T
    forecast = []
    for j in range(test_len):
        forecast.append(avg)
    return prediction, forecast


# RESIDUAL CALCULATION FOR AVERAGE METHOD
def residual_avg(train_s,prediction):
    predn = []
    for i in range(len(prediction)):
        predn.append(prediction[i]/(i+1))
    # print(predn)
    residual = []
    for i in range(len(train_s)):
        value = train_s.iloc[i] - predn[i-1]
        residual.append(value)
    residual.pop(0)
    return residual


# NAIVE METHOD
def naive_method(train_s, test_s):
    T = len(train_s)
    test_len = len(test_s)
    prediction = []
    for i in range(T):
        prediction.append(train_s.iloc[i - 1])
    # prediction.pop(0)
    # the 0th value is taken as the last obs in train set if pop is not done
    # for calculations, we must drop this value
    forecast = []
    for j in range(test_len):
        forecast.append(train_s.iloc[-1])
    return prediction, forecast


# DRIFT METHOD
def drift_method(train_s,test_s):
    T = len(train_s)
    prediction = []
    for i in range(1,T+1):
        if i == 1:
            prediction.append(train_s.iloc[0])
        else:
            num = (train_s.iloc[i-1] - train_s.iloc[0])/(i-1)
            num1 = train_s.iloc[i-1] + num
            prediction.append(num1)
    forecast = []
    test_len = len(test_s)
    y_T = train_s.iloc[-1]
    y_1 = train_s.iloc[0]
    # print(y_T,y_1)
    for h in range(test_len):
        h = h + 1
        num2 = y_T + (h * ((y_T-y_1)/(T - 1)))
        forecast.append(num2)
    return prediction, forecast


# SES METHOD
def ses_method(train_s, test_s, alpha):
    T = len(train_s)
    prediction = []
    for i in range(1,T+1):
        if i == 1:
            prediction.append(train_s.iloc[0])
        else:
            val1 = alpha*train_s.iloc[i-1]
            val2 = (1-alpha)*(prediction[-1])
            num = val1 + val2
            prediction.append(num)
    test_len = len(test_s)
    forecast = []
    for i in range(test_len):
        forecast.append(prediction[-1])
    return prediction, forecast


# RESIDUAL
def residual(train_s, prediction):
    residual = []
    T = len(train_s)
    for i in range(T):
        error = train_s.iloc[i]-prediction[i]
        residual.append(error)
    residual.pop(0)
    return residual


# FORECAST ERROR
def forecast_err(test_s, forec):
    f_error = []
    T = len(test_s)
    for i in range(T):
        error = test_s.iloc[i]-forec.iloc[i]
        f_error.append(error)
    return f_error


# MSE CALCULATIONS
def mse(pred_err, forec_err):
    mse_f = []
    mse_p = []
    for i in pred_err:
        mse_p.append(i**2)
    for j in forec_err:
        mse_f.append(j**2)
    return np.mean(mse_p), np.mean(mse_f)


# VARIANCE CALCULATION
def var(pred_err, forec_err):
    return np.var(pred_err), np.var(forec_err)


# AUTOCORRELATION FUNCTION
# Calculates tau value
def cal_acf(x,tau):
    x_bar = np.mean(x)
    den = 0
    num = 0
    for ele in range(len(x)):
        d = (x[ele] - x_bar)**2
        den += d
    for ele in range(tau,len(x)):
        n = (x[ele]-x_bar)*(x[ele-tau]-x_bar)
        num += n
    tau_n = num/den
    return tau_n


# Calculates the acf values
def acf_values(x, lag):
    tau_list = []
    for tau in range(len(lag)):
        tau_list.append(cal_acf(x,tau))
    # tau_list.pop(0)
    new_tau = []
    # print(tau_list)
    for each in tau_list:
        new_val = each**2
        new_tau.append(new_val)
    # print('The values for tau are as follows:')
    return new_tau


# Q VALUES
def Q_val(train_s, rk_vals):
    T = len(train_s)
    value = 0
    for val in rk_vals[1:]:
        value += (val**2)
    # print(value)
    Q = T*value
    return Q


# CORRELATION COEFFICIENT for dataset
def correlation_coefficient_cal_data(x,y):
    x_bar = np.mean(x)
    y_bar = y.mean()
    num = float(np.sum((x-x_bar)*(y-y_bar)))
    den1 = float(np.sqrt(np.sum((x-x_bar)**2)))
    den2 = float(np.sqrt(np.sum((y-y_bar)**2)))
    den = np.floor(den1*den2)
    r = num/den
    # print(num, den)
    return r


# CORRELATION COEFFICIENT for numpy array
def correlation_coefficient_cal1(x,y):
    a = np.matmul(x-np.mean(x),y-np.mean(y))
    b = np.sqrt(np.sum((x-np.mean(x))**2))
    c = np.sqrt(np.sum((y-np.mean(y))**2))
    corr_coeff = a / (b*c)
    return round(corr_coeff,2)


# BASIC PLOT FUNCTION
def plot_func(train_var1, test_var1, forecast_var1, train_var2, test_var2, forecast_var2, a=''):
    plt.figure()
    plt.plot(train_var1,train_var2, label='Training set')
    plt.plot(test_var1,test_var2, label='Testing set')
    plt.plot(forecast_var1, forecast_var2, label='h-step Forecast')
    plt.legend()
    plt.title('Plot for {} Method'.format(a))
    plt.ylabel('Data')
    plt.xlabel('Year')
    return plt.show()


# BASIC PLOT FUNCTION FOR SES
def plot_func_ses(train_var1, test_var1, forecast_var1, train_var2, test_var2, forecast_var2, alpha, a=''):
    plt.figure()
    plt.plot(train_var1,train_var2, label='Training set')
    plt.plot(test_var1,test_var2, label='Testing set')
    plt.plot(forecast_var1, forecast_var2, label='h-step Forecast')
    plt.legend()
    plt.title('Plot for {} Method when Alpha is {}'.format(a,alpha))
    plt.ylabel('Data')
    plt.xlabel('Year')
    return plt.show()


# PLOTTING DIFFERENT VALUES OF ALPHA - SES
def ses_plots(train_s, test_s, alpha, a='', b='',c=''):
    ses_pred, ses_forec = ses_method(train_s[a],test_s[a],alpha)
    final_predn = dataframe_create(ses_pred, train_s, b, a)
    final_forec = dataframe_create(ses_forec, test_s, b, a)
    return plot_func_ses(train_s[b],test_s[b],final_forec[b],train_s[a],test_s[a],final_forec[a],alpha,c)


# FUNCTION CALL FOR SES METHOD
def call_ses(train_s, test_s, alpha,lag, a='', b='', c=''):
    predn, forec = ses_method(train_s[a], test_s[a], alpha)
    final_predn = dataframe_create(predn, train_s, b, a)
    final_forec = dataframe_create(forec, test_s, b, a)
    plot_func_ses(train_s[b],test_s[b],final_forec[b],train_s[a],test_s[a],final_forec[a],alpha,c)
    residuals = residual(train_s[a], final_predn[a])
    forec_error = forecast_err(test_s[a], final_forec[a])
    mse_p, mse_f = mse(residuals, forec_error) # add another list in the function when needed
    var_p, var_f = var(residuals, forec_error)
    residual_acf = acf_values(residuals, lag)
    Q = Q_val(train_s, residual_acf)
    acf_plot(residual_acf, c)
    return final_predn, final_forec, residuals, forec_error, mse_p, mse_f, var_p, var_f, residual_acf, Q


# FUNCTION CALL FOR AVERAGE METHOD
def call_avg(train_s, test_s,lag, a='', b='', c=''):
    predn, forec = avg_method(train_s[a], test_s[a])
    final_predn = dataframe_create(predn, train_s, b, a)
    final_forec = dataframe_create(forec, test_s, b, a)
    plot_func(train_s[b],test_s[b],final_forec[b],train_s[a],test_s[a],final_forec[a],c)
    residuals = residual_avg(train_s[a], final_predn[a])
    forec_error = forecast_err(test_s[a], final_forec[a])
    mse_p, mse_f = mse(residuals, forec_error)
    var_p, var_f = var(residuals, forec_error)
    residual_acf = acf_values(residuals, lag)
    Q = Q_val(train_s, residual_acf)
    acf_plot(residual_acf, c)
    return final_predn, final_forec, residuals, forec_error, mse_p, mse_f, var_p, var_f, residual_acf, Q


# FUNCTION CALL FOR NAIVE METHOD
def call_naive(train_s, test_s, lag, a='', b='', c=''):
    predn, forec = naive_method(train_s[a], test_s[a])
    final_predn = dataframe_create(predn, train_s, b, a)
    final_forec = dataframe_create(forec, test_s, b, a)
    plot_func(train_s[b],test_s[b],final_forec[b],train_s[a],test_s[a],final_forec[a],c)
    residuals = residual(train_s[a], final_predn[a])
    forec_error = forecast_err(test_s[a], final_forec[a])
    mse_p, mse_f = mse(residuals, forec_error)
    var_p, var_f = var(residuals, forec_error)
    residual_acf = acf_values(residuals, lag)
    Q = Q_val(train_s, residual_acf)
    acf_plot(residual_acf, c)
    return final_predn, final_forec, residuals, forec_error, mse_p, mse_f, var_p, var_f, residual_acf, Q


# FUNCTION CALL FOR DRIFT METHOD
def call_drift(train_s, test_s, lag, a='', b='', c=''):
    predn, forec = drift_method(train_s[a], test_s[a])
    final_predn = dataframe_create(predn, train_s, b, a)
    final_forec = dataframe_create(forec, test_s, b, a)
    plot_func(train_s[b],test_s[b],final_forec[b],train_s[a],test_s[a],final_forec[a],c)
    residuals = residual(train_s[a], final_predn[a])
    forec_error = forecast_err(test_s[a], final_forec[a])
    mse_p, mse_f = mse(residuals, forec_error)
    var_p, var_f = var(residuals, forec_error)
    residual_acf = acf_values(residuals, lag)
    Q = Q_val(train_s, residual_acf)
    acf_plot(residual_acf, c)
    return final_predn, final_forec, residuals, forec_error, mse_p, mse_f, var_p, var_f, residual_acf, Q


# FUNCTION CALL FOR HOLT'S SEASONAL METHOD
def call_holt_s(train_s,test_s,lag, a='',b='',c=''):
    prediction = ets.ExponentialSmoothing(train_s[a], trend='additive', seasonal='additive', seasonal_periods=1440,damped_trend=True).fit()
    forecast = prediction.forecast(steps=len(test_s[a]))
    yt = prediction.fittedvalues
    final_predn = dataframe_create(yt,train_s,b,a)
    final_forec = dataframe_create(forecast,test_s,b,a)
    plot_func(train_s[b],test_s[b],final_forec[b],train_s[a],test_s[a],final_forec[a],c)
    residuals = residual_avg(train_s[a], final_predn[a])
    forec_error = forecast_err(test_s[a], final_forec[a])
    mse_p, mse_f = mse(residuals, forec_error)
    var_p, var_f = var(residuals, forec_error)
    residual_acf = acf_values(residuals, lag)
    Q = Q_val(train_s, residual_acf)
    acf_plot(residual_acf, c)
    return prediction, final_forec, residuals, forec_error, mse_p, mse_f, var_p, var_f, residual_acf, Q


# FUNCTION CALL FOR HOLT'S LINEAR METHOD
def call_holt_l(train_s,test_s,lag, a='',b='',c=''):
    prediction = ets.ExponentialSmoothing(train_s[a], trend='additive', seasonal=None, damped=True).fit()
    # trend='multiplicative', seasonal=None, damped=True
    # trend='additive', seasonal=None, damped=True
    forecast = prediction.forecast(steps=len(test_s[a]))
    yt = prediction.fittedvalues
    final_predn = dataframe_create(yt,train_s,b,a)
    final_forec = dataframe_create(forecast,test_s,b,a)
    plot_func(train_s[b],test_s[b],final_forec[b],train_s[a],test_s[a],final_forec[a],c)
    residuals = residual_avg(train_s[a], final_predn[a])
    forec_error = forecast_err(test_s[a], final_forec[a])
    mse_p, mse_f = mse(residuals, forec_error)
    var_p, var_f = var(residuals, forec_error)
    residual_acf = acf_values(residuals, lag)
    Q = Q_val(train_s, residual_acf)
    acf_plot(residual_acf, c)
    return prediction, final_forec, residuals, forec_error, mse_p, mse_f, var_p, var_f, residual_acf, Q


# FUNCTION FOR NORMAL EQUATION (LINEAR REGRESSION)
def normal_eq(y_train,x_train):
    x_transpose = np.transpose(x_train)
    e1 = np.dot(x_transpose,x_train)
    inverse = np.linalg.inv(e1)
    e2 = np.dot(x_transpose,y_train)
    b_hat = np.dot(inverse,e2)
    return b_hat


# FUNCTION TO CALCULATE VARIANCE FOR LINEAR REGRESSION
def var_linreg(error_arr, num_features):
    T = len(error_arr)
    k = num_features
    e1 = 1/(T-k-1)
    val = np.sum(np.square(error_arr))
    sigma_e = np.sqrt(e1*val)
    return sigma_e


# FUNCTION FOR BACKWARD STEPWISE REGRESSION
def backward_reg(x_train, y_train, drop_feature=''):
    x_train = x_train.drop(columns=drop_feature)
    X = sm.add_constant(x_train)
    model = sm.OLS(y_train, X).fit()
    return model, x_train


# FUNCTION FOR FORWARD STEPWISE REGRESSION
def forward_reg(y_train,x_train, x_trainf,feature,name=''):
    x_trainf[name] = x_train[feature]
    X = sm.add_constant(x_trainf)
    model = sm.OLS(y_train, X).fit()
    return model, x_trainf


# FUNCTION FOR MSE MULTIPLE LINEAR REGRESSION
def mse_mlr(pred_err):
    mse_p = []
    for i in pred_err:
        mse_p.append(i**2)
    return np.mean(mse_p)


# CORRELATION COEFFICIENT MULTIPLE LINEAR REGRESSION
def correlation_coefficient_mlr(x,y):
    x_bar = np.mean(x)
    y_bar = y.mean()
    num = float(np.sum((x-x_bar)*(y-y_bar)))
    den1 = float(np.sqrt(np.sum((x-x_bar)**2)))
    den2 = float(np.sqrt(np.sum((y-y_bar)**2)))
    den = np.floor(den1*den2)
    r = num/den
    # print(num, den)
    return r

# FUNCTION TO CALCULATE STRENGTH
def strength_stats(trend, seasonal, resid):
    ft = np.maximum(0,1 - (np.var(np.array(resid))/(np.var(np.array(trend+resid)))))
    fs = np.maximum(0,1 - (np.var(np.array(resid))/(np.var(np.array(seasonal+resid)))))
    return ft, fs


# FUNCTION FOR PARTIAL CORRELATION
def partial_corr(ab, ac, bc):
    r_partial = (ab-ac*bc)/(np.sqrt(1-ac**2)*np.sqrt(1-bc**2))
    return r_partial


# FUNCTION TO CALCULATE t0
def t_test_pc(r_ab_c, n, k):
    t0 = r_ab_c*np.sqrt((n-2-k)/(1-r_ab_c**2))
    return t0


# FUNCTION TO CALCULATE PHI - GPAC
def phical(acfcal,na,nb):
    den = []
    k = na
    j = nb

    for a in range(k):
        den.append([])
        for b in range(k):
            den[a].append(acfcal[np.abs(j + b)])
        j = j - 1

    j = nb
    num = den[:k - 1]
    num.append([])
    for a in range(k):
        num[k - 1].append(acfcal[j + a + 1])

    det_num = round(np.linalg.det(num),5)
    det_den = round(np.linalg.det(den),5)

    if det_den == 0:
        return float('inf')

    phi = det_num / det_den

    return round(phi,3)


# FUCNCTION TO CALCULATE GPAC AND PLOT SNS PLOT
def gpac_cal(acfcal,k,j):
    phi = []
    for b in range(j):
        phi.append([])
        for a in range(1, k+1):
            phi[b].append(phical(acfcal, a, b))

    gpac = np.array(phi).reshape(j, k)
    gpac_df = pd.DataFrame(gpac)
    cols = np.arange(1, k+1)
    gpac_df.columns = cols
    print(gpac_df)
    print()

    sns.heatmap(gpac_df, annot=True)
    plt.xlabel('AR process(k)')
    plt.ylabel('MA process(j)')
    plt.title('Generalized Partial Autocorrelation (GPAC) table')
    plt.show()


# FUNCTION FOR CONFIDENCE INTERVAL
def confidence_interval(cov, na, nb, params):
    print('\n Confidence Interval for Estimated parameters')
    for i in range(na):
        upper = params[i] + 2 * np.sqrt(cov[i][i])
        lower = params[i] - 2 * np.sqrt(cov[i][i])
        print(lower, '< a{} <'.format(i+1), upper)

    for j in range(nb):
        upper = params[na+j] + 2 * np.sqrt(cov[na+j][na+j])
        lower = params[na+j] - 2 * np.sqrt(cov[na+j][na+j])
        print(lower, '< b{} <'.format(j + 1), upper)


# FUNCTION FOR ZEROS AND POLES
def zeros_and_poles(est_params, na, nb):
    p_ceoff = [1] + list(est_params[:na])
    z_coeff = [1] + list(est_params[na:])
    poles = np.roots(p_ceoff)
    zeros = np.roots(z_coeff)
    print('\nZeros : ',zeros)
    print('Poles : ',poles)
    return zeros, poles


# Chi-square test
def chi_square_test(Q, lags, na, nb, alpha=0.01):
    dof = lags - na - nb
    chi_critical = chi2.isf(alpha, df=dof)

    if Q < chi_critical:
        print(f"The residual is white and the estimated order is n_a = {na} and n_b = {nb}")
    else:
        print(f"The residual is not white with n_a={na} and n_b={nb}")

    return Q < chi_critical


# FUNCTION TO CREATE DATAFRAME
def dataframe_create_arma(prediction, test_set, a=''):
    forecast1 = pd.DataFrame({a: prediction}, index=test_set.index)
    return forecast1

# RESIDUAL ARMA
def residual_arma(train_s, prediction):
    residual = []
    T = len(train_s)
    for i in range(T):
        error = train_s.iloc[i]-prediction[i]
        residual.append(error)
    residual.pop(0)
    return residual


# FORECAST ERROR ARMA
def forecast_err_arma(test_s, forec):
    f_error = []
    T = len(test_s)
    for i in range(T):
        error = test_s.iloc[i]-forec[i]
        f_error.append(error)
    return f_error
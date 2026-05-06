import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
import warnings
from typing import Dict, Any, Union

def test_stationarity(
    data: Union[np.ndarray, pd.Series], 
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Performs ADF and KPSS tests to determine time series stationarity.
    """
    series = np.asarray(data)
    adf_result = adfuller(series, autolag='AIC')
    adf_pvalue = float(adf_result[1])
    adf_stationary = adf_pvalue < significance_level
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_pvalue = float(kpss_result[1])
    kpss_stationary = kpss_pvalue >= significance_level
    
    if adf_stationary and kpss_stationary:
        status = 'Strictly Stationary'
    elif not adf_stationary and not kpss_stationary:
        status = 'Strictly Non-Stationary'
    elif not adf_stationary and kpss_stationary:
        status = 'Trend Stationary (Needs Detrending)'
    else:
        status = 'Difference Stationary (Needs Differencing)'
        
    return {
        'ADF_pvalue': adf_pvalue,
        'KPSS_pvalue': kpss_pvalue,
        'Is_Stationary_ADF': adf_stationary,
        'Is_Stationary_KPSS': kpss_stationary,
        'Conclusion': status
    }

def test_residual_diagnostics(
    residuals: Union[np.ndarray, pd.Series], 
    lags: int = 10,
    significance_level: float = 0.05
) -> Dict[str, Any]:
    """
    Tests model residuals for temporal assumptions: Autocorrelation (Ljung-Box), 
    Volatility Clustering (ARCH-LM), and Normality (Jarque-Bera).
    """
    resids = np.asarray(residuals)
    
    lb_df = acorr_ljungbox(resids, lags=[lags], return_df=True)
    lb_pvalue = float(lb_df['lb_pvalue'].iloc[0])
    has_autocorr = lb_pvalue < significance_level
    
    arch_stat, arch_pvalue, _, _ = het_arch(resids, nlags=lags)
    arch_pvalue = float(arch_pvalue)
    has_arch_effects = arch_pvalue < significance_level
    
    jb_stat, jb_pvalue = stats.jarque_bera(resids)
    jb_pvalue = float(jb_pvalue)
    is_normal = jb_pvalue >= significance_level
    
    return {
        'Ljung_Box_pvalue': lb_pvalue,
        'Has_Autocorrelation': has_autocorr,
        'ARCH_LM_pvalue': arch_pvalue,
        'Has_Volatility_Clustering': has_arch_effects,
        'Jarque_Bera_pvalue': jb_pvalue,
        'Is_Normal': is_normal
    }
    
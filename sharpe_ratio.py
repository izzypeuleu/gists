import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculates the Sharpe Ratio for a given set of returns.
    
    Parameters:
    returns (pandas.Series): Series of returns
    risk_free_rate (float): Risk-free rate of return (default 0.02)
    
    Returns:
    float: Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def sharpe_ratio_weighted_risk_score(returns, existing_risk_score, sharpe_ratio_weight=0.5):
    """
    Calculates a weighted risk score incorporating the Sharpe Ratio.
    
    Parameters:
    returns (pandas.Series): Series of returns
    existing_risk_score (float): Existing risk score
    sharpe_ratio_weight (float): Weight to assign to the Sharpe Ratio (default 0.5)
    
    Returns:
    float: Updated risk score
    """
    sharpe_ratio = calculate_sharpe_ratio(returns)
    return (1 - sharpe_ratio_weight) * existing_risk_score + sharpe_ratio_weight * sharpe_ratio

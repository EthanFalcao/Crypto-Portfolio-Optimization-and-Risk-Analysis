"""Mean-Variance portfolio optimization (Markowitz).

The idea: pick a weight for each coin (how much of the portfolio to put into
it) that gives the best tradeoff between expected return and risk. "Risk" here
just means variance - how much the portfolio's value bounces around.

Rules we enforce:
- no shorting (every weight is >= 0)
- fully invested (all the weights add up to 1)
- no single coin can be more than max_weight of the portfolio (this is also
  our "max exposure" risk rule - see the README)
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src import config


def portfolio_return(weights, expected_returns):
    return np.dot(weights, expected_returns)


def portfolio_variance(weights, cov_matrix):
    return np.dot(weights, np.dot(cov_matrix, weights))


def mean_variance_weights(expected_returns, cov_matrix, risk_aversion=config.RISK_AVERSION,
                           max_weight=config.MAX_ASSET_WEIGHT):
    """
    expected_returns: pandas Series, one predicted return per coin
    cov_matrix: pandas DataFrame, covariance between every pair of coins
    Returns a pandas Series of weights (same coins), adding up to 1.
    """
    coins = expected_returns.index
    num_coins = len(coins)
    mu = expected_returns.values
    sigma = cov_matrix.loc[coins, coins].values

    # scipy's minimize() only knows how to minimize, so to maximize
    # (return - risk penalty) we minimize its negative instead.
    def score_to_minimize(weights):
        expected_gain = portfolio_return(weights, mu)
        risk_penalty = (risk_aversion / 2) * portfolio_variance(weights, sigma)
        return -(expected_gain - risk_penalty)

    def weights_must_sum_to_one(weights):
        return weights.sum() - 1

    constraints = [{"type": "eq", "fun": weights_must_sum_to_one}]
    bounds = [(0, max_weight) for _ in range(num_coins)]
    equal_weight_guess = np.array([1 / num_coins for _ in range(num_coins)])

    result = minimize(
        score_to_minimize,
        equal_weight_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if result.success:
        best_weights = result.x
    else:
        # Rare (e.g. a weird covariance matrix). Fall back to equal weight
        # rather than return a broken/failed result.
        best_weights = equal_weight_guess

    return pd.Series(best_weights, index=coins)

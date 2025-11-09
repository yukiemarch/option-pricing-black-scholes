import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def _d1_d2(S, K, T, r, sigma):
    """
    Helper function to compute d1 and d2 in Black–Scholes.
    S : spot price
    K : strike
    T : time to maturity (in years)
    r : risk-free rate (annualized, continuously compounded)
    sigma : volatility (annualized)
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Black–Scholes price for a European call or put.

    option_type: "call" or "put"
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price


def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Compute main Greeks for a European option under Black–Scholes.
    Returns a dict with Delta, Gamma, Vega, Theta, Rho.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)

    if option_type == "call":
        delta = cdf_d1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * cdf_d2)
        rho = K * T * np.exp(-r * T) * cdf_d2
    elif option_type == "put":
        delta = cdf_d1 - 1
        theta = (- (S * pdf_d1 * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * pdf_d1 * np.sqrt(T)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega / 100.0,   # per 1% vol
        "theta": theta / 365.0, # per day
        "rho": rho / 100.0      # per 1% rate change
    }


def implied_volatility(target_price, S, K, T, r, option_type="call",
                       sigma_lower=1e-6, sigma_upper=5.0):
    """
    Solve for implied volatility given a market price.
    Uses Brent's root-finding method.
    """
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - target_price

    try:
        iv = brentq(objective, sigma_lower, sigma_upper)
    except ValueError:
        # no root found in the interval
        iv = np.nan
    return iv

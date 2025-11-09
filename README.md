# Option Pricing and Greeks in Python (Black–Scholes)

Black–Scholes option pricing and Greeks in Python.

It implements the Black–Scholes model for European options in Python and computes the main Greeks (Delta, Gamma, Vega, Theta, Rho), plus an implied volatility solver.

## Features

- Black–Scholes pricing for European calls and puts  
- Greeks:
  - Delta
  - Gamma
  - Vega (per 1% volatility change)
  - Theta (per day)
  - Rho (per 1% rate change)
- Implied volatility solver using Brent's method (`scipy.optimize.brentq`)

## Project Structure

option-pricing-black-scholes/
- README.md
- requirements.txt
- src/
  - bs_pricer.py

## Example usage
```
from src.bs_pricer import black_scholes_price, black_scholes_greeks

S, K, T, r, sigma = 100, 100, 1.0, 0.02, 0.2

call_price = black_scholes_price(S, K, T, r, sigma, "call")
greeks = black_scholes_greeks(S, K, T, r, sigma, "call")

print("Call price:", call_price)
print("Greeks:", greeks)
```

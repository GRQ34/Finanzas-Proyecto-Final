import yfinance as yf
from scipy.optimize import minimize
import numpy as np
import os

tickers = yf.Tickers('ASML NEE PH SBGSF')
hist = tickers.history(start='2021-01-01',end='2024-12-31')
print('Data:')
print(hist)

adj_close = hist['Close']
print('Adj Close:')
print(adj_close)


if os.path.exists('data.xlsx'):
    os.remove('data.xlsx')
adj_close.to_excel('data.xlsx')

adj_close_values = adj_close.values
print('Adj Close values:')
print(np.round(adj_close_values,4))
print(adj_close_values.shape)


R = np.log(adj_close_values[1:] / adj_close_values[:-1])
print('R:')
print(np.round(R*100, 4), '%')




RE = np.mean(R, axis=0)*252
RI = np.std(R, axis=0)*np.sqrt(252)
print('RE:', np.round(RE*100, 4), '%')
print('RI:', np.round(RI*100, 4), '%')


Sharpes = RE / RI
print('Sharpes:', np.round(Sharpes, 4))

S = np.cov(R, rowvar=False)
print('S:')
print(S*100, '%')


corr = np.corrcoef(R, rowvar=False)
print('Correlation:')
print(corr)

n_assets = R.shape[1]
weights = np.ones(n_assets) / n_assets
print('Weights:', np.round(weights, 4))
print('Sum of weights:', np.sum(weights))


def rep(w,r):
    return w @ np.transpose(r)

def varp(w,s):
    return w @ s @ np.transpose(w)


ReP = rep(weights, RE)
varP = varp(weights, S)
RiP = np.sqrt(varP)
SharpeP = ReP / RiP
print('Portafolio pre-optimización:')
print('ReP:', round(ReP*100, 4), '%')
print('varP:', round(varP, 4))
print('RiP:', round(RiP*100, 4), '%')
print('SharpeP:', round(SharpeP, 4))


def constr(w):
    return np.sum(w) - 1

num_assets = 4
bounds = [(0, None)] * num_assets

constraints = [{'type': 'eq', 'fun': constr}]

res = minimize(
    fun=lambda w: varp(w, S),
    x0=weights,
    method='SLSQP',
    bounds=bounds,
    constraints=constraints,
    options={'disp': False, 'maxiter': 1000, 'ftol': 1e-12}
)

print('Pesos optimizados:')
print(np.round(res.x*100, 4), '%')


ReP = rep(res.x, RE)
varP = varp(res.x, S)
RiP = np.sqrt(varP)
SharpeP = ReP / RiP
print('Portafolio post-optimización:')
print('ReP:', round(ReP*100, 4), '%')
print('varP:', round(varP, 4))
print('RiP:', round(RiP*100, 4), '%')
print('SharpeP:', round(SharpeP, 4))



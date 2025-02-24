import yfinance as yf
from scipy.optimize import minimize
import numpy as np
import pandas as pd
from itertools import combinations


# Lista de empresas a evaluar
tickers_list = ['NEE', 'AMAT', 'ROK', 'ASML', 'SIEGY', 'SBGSF', 'EMR', 'HON', 'PH', 'ETN']

# Descargar datos históricos
tickers = yf.Tickers(" ".join(tickers_list))
hist = tickers.history(start='2021-01-01', end='2024-12-31')
adj_close = hist['Close'].dropna(axis=1)  # Eliminar valores NaN


# Función para calcular métricas del portafolio
def calcular_metricas(adj_close_values):
    R = np.log(adj_close_values[1:] / adj_close_values[:-1])  # Retornos logarítmicos
    RE = np.mean(R, axis=0) * 252  # Retorno esperado anualizado
    RI = np.std(R, axis=0) * np.sqrt(252)  # Riesgo anualizado
    S = np.cov(R, rowvar=False)  # Matriz de covarianza
    corr = np.corrcoef(R, rowvar=False)  # Matriz de correlación
    return RE, RI, S, corr

# Obtener métricas generales
adj_close_values = adj_close.values
RE, RI, S, correlation_matrix = calcular_metricas(adj_close_values)

# Evaluar todas las combinaciones posibles de 4 activos
mejores_portafolios = []
for subset in combinations(adj_close.columns, 4):
    indices = [adj_close.columns.get_loc(ticker) for ticker in subset]
    corr_submatrix = correlation_matrix[np.ix_(indices, indices)]

    # Filtrar combinaciones con correlación > 0.5
    if np.any(np.abs(np.triu(corr_submatrix, k=1)) > 0.5):
        continue

    # Extraer datos de la combinación aceptable
    RE_sub = RE[indices]
    S_sub = S[np.ix_(indices, indices)]
    weights = np.ones(4) / 4  # Pesos iniciales iguales

    # Definir restricciones y límites
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0, None)] * 4

    # Optimizar el portafolio
    res = minimize(lambda w: w @ S_sub @ w.T, x0=weights, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False, 'maxiter': 1000, 'ftol': 1e-12})

    if res.success:
        # Calcular métricas del portafolio optimizado
        ReP = res.x @ RE_sub.T
        varP = res.x @ S_sub @ res.x.T
        RiP = np.sqrt(varP)
        SharpeP = ReP / RiP

        # Guardar resultados
        mejores_portafolios.append({
            "Activos": subset,
            "ReP (%)": round(ReP * 100, 4),
            "RiP (%)": round(RiP * 100, 4),
            "SharpeP": round(SharpeP, 4)
        })

# Ordenar por el mejor Sharpe Ratio
mejores_portafolios = sorted(mejores_portafolios, key=lambda x: x["SharpeP"], reverse=True)

# Convertir a DataFrame y mostrar los resultados
df_resultados = pd.DataFrame(mejores_portafolios)
print(df_resultados)

# Guardar en un archivo Excel
df_resultados.to_excel("mejores_portafolios.xlsx", index=False)
print("\nResultados guardados en 'mejores_portafolios.xlsx'")



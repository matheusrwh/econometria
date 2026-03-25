import numpy as np
from scipy.stats import t as t_dist, f as f_dist

class OLS:
    """
    Estimador de mínimos quadrados ordinários (OLS).

    Parâmetros
    ----------
    Y: vetor (n, 1)
       Variável dependente.
    X: matriz (n, k) ou (n, 1)
       Variáveis explicativas.
    add_constant: bool, default=True
       Se True, adiciona uma coluna de 1's em X para servir de intercepto.
    """

    def __init__(self, Y, X, add_constant: bool = True):
        self.Y = np.asarray(Y, dtype=float)
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if add_constant:
            ones = np.ones((X.shape[0], 1))
            self.X = np.hstack((ones, X))
        else:
            self.X = X
        
        self._fitted = False

    def _estimate_beta(self):
        """
        Resolve o problema de minimização para obter os coeficientes beta.
        
        beta_hat = (X'X)^(-1) X'Y.
        """

        XtX = self.X.T @ self.X                         # X'X (k, k)
        XtY = self.X.T @ self.Y                         # X'Y (k, 1)
        self.beta_hat = np.linalg.solve(XtX, XtY)       # beta_hat (k, 1)
        self.XtX_inv = np.linalg.inv(XtX)               # (X'X)^(-1) para variância dos coeficientes

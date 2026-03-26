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
    
    # -----------------------------------------------------
    # 1. Estima os coeficientes beta usando a fórmula de OLS.
    # -----------------------------------------------------
    def _estimate_beta(self):
        """
        Resolve o problema de minimização para obter os coeficientes beta.
        
        beta_hat = (X'X)^(-1) X'Y.
        """

        XtX = self.X.T @ self.X                         # X'X (k, k)
        XtY = self.X.T @ self.Y                         # X'Y (k, 1)
        self.beta_hat = np.linalg.solve(XtX, XtY)       # beta_hat (k, 1)
        self.XtX_inv = np.linalg.inv(XtX)               # (X'X)^(-1) para variância dos coeficientes

    # -----------------------------------------------------
    # 2. Resíduos e variância.
    # -----------------------------------------------------
    def _compute_residuals(self):
        """
        Calcula diretamente o y_hat e os resíduos Y - y_hat.
        """

        self.y_hat = self.X @ self.beta_hat             # Previsões (n, 1)
        self.residuals = self.Y - self.y_hat            # Resíduos (n, 1)
    
    def _estimate_sigma2(self):
        """
        Estima a sigma^2 a partir da soma dos quadrados dos resíduos.
        """

        n, k = self.X.shape
        SSR = self.residuals @ self.residuals
        self.sigma2_hat = SSR / (n - k)                # Sigma^2 estimado (escalar)
        self.SSR = SSR                                 # Soma dos quadrados dos resíduos (escalar)

        self.n = n
        self.k = k

    # -----------------------------------------------------
    # 3. Matriz de variância-covariância e inferência.
    # -----------------------------------------------------
    def _inference(self):
        """
        Calcula a matriz de variância-covariância dos coeficientes, estatísticas t e p-valores bicaudais.
        """

        n, k = self.X.shape
        self.cov_beta = self.sigma2_hat * self.XtX_inv                   # Matriz de variância-covariância (k, k)
        self.se = np.sqrt(np.diag(self.cov_beta))                        # Erros padrão dos coeficientes (k,)
        self.t_stats = self.beta_hat / self.se                           # Estatísticas t (k, 1)

        self.p_values = 2 * t_dist.sf(np.abs(self.t_stats), df = n - k)  # P-valores bicaudais (k, 1)

    # -----------------------------------------------------
    # FINAL - Fit do modelo - Interface pública.
    # -----------------------------------------------------
    def fit(self):
        """
        Executa todas as etapas de estimação.
        """

        self._estimate_beta()
        self._compute_residuals()
        self._estimate_sigma2()
        self._inference()
        self._fitted = True
        return self
    
    def summary(self):
        """
        Imprime os resultados da regressão.
        """

        if not self._fitted:
            raise RuntimeError("Você precisa ajustar o modelo antes de chamar summary().")

        n, k = self.n, self.k
        sep = "=" * 65

        print(sep)
        print(f"{'Resultados da regressão por OLS':^65}")
        print(sep)
        print(f" Observações: {n:>10}               Parâmetros: {k:>10}")
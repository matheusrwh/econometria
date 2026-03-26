from tkinter.font import names

from distro import name
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
    # 4. Inferência de erros robustos à heterocedasticidade.
    # -----------------------------------------------------
    def _inference_robust(self):
        """
        Calcula a matriz de variância-covariância robusta de White com correção de graus de liberdade.
        Esse comando é equivalente ao vce(robust) do Stata.

        Cov_robust(beta_hat) = (X'X)^{-1} B_HC1 (X'X)^{-1}
        onde B_HC1 = (n / (n-k)) * sum_i( e_i^2 * x_i x_i' )
                   = (n / (n-k)) * X' diag(e^2) X
        """

        n, k = self.X.shape
        e = self.residuals.flatten()

        Xe = self.X * e[:, None]
        meat = (n / (n - k)) * (Xe.T @ Xe)                               # Matriz "meat" de White (k, k)

        self.cov_beta_robust = self.XtX_inv @ meat @ self.XtX_inv        # Matriz de variância-covariância robusta (k, k)
        self.se_robust = np.sqrt(np.diag(self.cov_beta_robust))
        self.t_stats_robust = self.beta_hat.flatten() / self.se_robust
        self.p_values_robust = 2 * t_dist.sf(np.abs(self.t_stats_robust), df = n - k)


    # -----------------------------------------------------
    # 5. Medidas de qualidade do ajuste e teste F.
    # -----------------------------------------------------
    def _goodness_of_fit(self):
        """
        R^2 = 1 - SSR/SST
        R^2 ajustado = 1 - (n-1)/(n-k) * (1 - R^2)
        F = [R^2 / (k-1)] / [(1 - R^2) / (n-k)]
        """

        n, k = self.X.shape
        SST = np.sum((self.Y - self.Y.mean()) ** 2)                                         # Soma total dos quadrados (escalar)

        self.r_squared = 1 - self.SSR / SST                                                 # R^2 (escalar)
        self.r_squared_adj = 1 - (n - 1) / (n - k) * (1 - self.r_squared)                   # R^2 ajustado (escalar)

        k_restr = k - 1
        if k_restr > 0:
            self.f_stat = (self.r_squared / k_restr) / ((1 - self.r_squared) / (n - k))     # Estatística F (escalar)
            self.f_pvalue = f_dist.sf(self.f_stat, dfn = k_restr, dfd = n - k)              # P-valor do teste F (escalar)
        else:
            self.f_stat = np.nan
            self.f_pvalue = np.nan
        
        self.n = n
        self.k = k

    # -----------------------------------------------------
    # FINAL - Fit do modelo - Interface pública.
    # -----------------------------------------------------
    def fit(self, robust: bool = False):
        """
        Executa todas as etapas de estimação.
        """

        self._estimate_beta()
        self._compute_residuals()
        self._estimate_sigma2()
        self._inference()
        if robust:
            self._inference_robust()
        self._goodness_of_fit()
        self._fitted = True
        self._robust = robust
        return self
    
    def summary(self):
        """Imprime tabela de resultados no estilo de pacotes econométricos."""
        if not self._fitted:
            raise RuntimeError("Execute .fit() antes de chamar .summary().")
 
        n, k = self.n, self.k
        sep = "=" * 65
 
        # Nomes dos coeficientes
        names = (["const"] if k > 1 else []) + [f"x{j}" for j in range(1, k)]

        # Seleciona os vetores corretos conforme o tipo de erro estimado
        se_use = self.se_robust       if self._robust else self.se
        t_use  = self.t_stats_robust  if self._robust else self.t_stats
        p_use  = self.p_values_robust if self._robust else self.p_values
        label  = "OLS, erros robustos"       if self._robust else "OLS"

        print(sep)
        print(f"{'Resultados da regressão - ' + label:^65}")
        print(sep)
        print(f"  Obs.:           {n:>10}       R²:            {self.r_squared:>10.4f}")
        print(f"  Parâmetros (k): {k:>10}       R² ajustado:   {self.r_squared_adj:>10.4f}")
        print(f"  Sigma² estimado:{self.sigma2_hat:>10.4f}       F ({k-1}, {n-k}):      {self.f_stat:>10.4f}")
        print(f"  {'':24}         p-value (F):   {self.f_pvalue:>10.4f}")
        print("-" * 65)
        print(f"  {'Coef.':<12} {'beta_hat':>10} {'std err':>10} {'t':>10} {'P>|t|':>10}")
        print("-" * 65)
        for name, b, se, t, p in zip(names, self.beta_hat.flatten(), se_use, t_use, p_use):
            sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
            print(f"  {name:<12} {b:>10.4f} {se:>10.4f} {t:>10.4f} {p:>10.4f} {sig}")

        print(sep)

        print("  Nível de signif.: *** 1%  ** 5%  * 10%")
        print(sep)

    def predict(self, X_new):
        """
        Gera previsões a partir do modelo ajustado.
        """

        if not self._fitted:
            raise RuntimeError("Execute .fit() antes de chamar .predict().")
        
        X_new = np.asarray(X_new, dtype=float)
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        if self.X.shape[1] > X_new.shape[1]:
            X_new = np.hstack((np.ones((X_new.shape[0], 1)), X_new))
        return X_new @ self.beta_hat

# -----------------------------------------------------
# Exemplo de uso.
# -----------------------------------------------------
if __name__ == "__main__":
    import wooldridge
    wage = wooldridge.data('wage1')

    Y = wage['wage']
    X = wage['educ']

    model = OLS(Y, X, add_constant=True)
    model = model.fit(robust = True)
    model.summary()
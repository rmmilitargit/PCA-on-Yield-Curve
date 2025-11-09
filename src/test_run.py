import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -----------------------------
# 1. SIMULATE YIELD CURVE DATA
# -----------------------------
np.random.seed(42)

tenors = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
n_obs = 250  # ~1 trading year

# base level + correlated random noise
base_curve = 0.03 + 0.002 * (tenors - tenors.mean())
cov_matrix = np.exp(-np.abs(np.subtract.outer(tenors, tenors)) / 2.0)
random_shocks = np.random.multivariate_normal(np.zeros(len(tenors)), cov_matrix, size=n_obs)
yields = base_curve + 0.005 * random_shocks  # simulated daily yield curves

yield_df = pd.DataFrame(yields, columns=[f"{t}Y" for t in tenors])
print("\nSample Yield Curves:\n", yield_df.head())

# -----------------------------
# 2. PCA ON YIELD CURVES
# -----------------------------
pca = PCA(n_components=3)
pca.fit(yield_df)

pc_scores = pca.transform(yield_df)
explained_var = pca.explained_variance_ratio_

print("\nExplained Variance by PCs:", np.round(explained_var, 4))
loadings = pd.DataFrame(pca.components_.T, index=yield_df.columns,
                        columns=['PC1 (Level)', 'PC2 (Slope)', 'PC3 (Curvature)'])
print("\nPCA Loadings:\n", loadings)

# -----------------------------
# 3. SAMPLE BOND PORTFOLIO
# -----------------------------
portfolio = pd.DataFrame({
    "Bond": ["B1", "B2", "B3"],
    "Maturity": [2, 5, 10],
    "Coupon": [0.03, 0.035, 0.04],
    "FaceValue": [100, 100, 100]
})

# simple PV calculation
def bond_price(y, maturity, coupon, face=100, freq=2):
    n = int(maturity * freq)
    c = coupon / freq * face
    times = np.arange(1, n + 1) / freq
    discount = np.exp(-y * times)
    pv = np.sum(c * discount) + face * np.exp(-y * maturity)
    return pv

# Key Rate Durations (small shock ±1bp per tenor)
def compute_krd(base_curve, tenors, bond):
    krd = []
    base_yield = np.interp(bond["Maturity"], tenors, base_curve)
    pv0 = bond_price(base_yield, bond["Maturity"], bond["Coupon"])
    bump = 0.0001  # 1bp
    for i, t in enumerate(tenors):
        bumped = base_curve.copy()
        bumped[i] += bump
        y_bumped = np.interp(bond["Maturity"], tenors, bumped)
        pv_bumped = bond_price(y_bumped, bond["Maturity"], bond["Coupon"])
        krd_i = (pv_bumped - pv0) / (pv0 * bump)
        krd.append(krd_i)
    return np.array(krd)

krd_matrix = np.vstack([compute_krd(base_curve, tenors, row) for _, row in portfolio.iterrows()])
krd_df = pd.DataFrame(krd_matrix, columns=[f"{t}Y" for t in tenors], index=portfolio["Bond"])
print("\nKey Rate Durations:\n", krd_df.round(4))

# -----------------------------
# 4. MAP KRDs TO PCA FACTORS
# -----------------------------
factor_exposure = krd_df.values @ loadings.values
factor_df = pd.DataFrame(factor_exposure,
                         columns=loadings.columns,
                         index=krd_df.index)
print("\nFactor Exposures (PCA-space):\n", factor_df.round(4))

# -----------------------------
# 5. SCENARIO ANALYSIS
# -----------------------------
# ±1σ shifts along each PCA
factor_stdev = np.sqrt(pca.explained_variance_)
scenarios = {"+1σ_PC1": factor_stdev[0],
             "-1σ_PC1": -factor_stdev[0],
             "+1σ_PC2": factor_stdev[1],
             "-1σ_PC2": -factor_stdev[1],
             "+1σ_PC3": factor_stdev[2],
             "-1σ_PC3": -factor_stdev[2]}

scenario_pnl = {}
for s_name, s_shift in scenarios.items():
    pnl = (factor_df * s_shift).sum(axis=1)
    scenario_pnl[s_name] = pnl

pnl_df = pd.DataFrame(scenario_pnl)
print("\nScenario P&L (approximate):\n", pnl_df.round(4))

# -----------------------------
# 6. VISUALIZATION
# -----------------------------
plt.figure(figsize=(8, 4))
plt.plot(loadings, marker='o')
plt.title("PCA Loadings on Yield Tenors")
plt.xlabel("Tenor")
plt.ylabel("Loading")
plt.grid(True)
plt.legend(loadings.columns)
plt.tight_layout()
plt.show()

print("\n✅ Analysis complete.")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ======================================================
# 1. SIMULATED HISTORICAL YIELD CURVES
# ======================================================
np.random.seed(42)
dates = pd.bdate_range("2020-01-01", "2024-12-31")
n = len(dates)
tenors = np.array([1, 2, 3, 5, 7, 10, 20, 30])
n_nodes = len(tenors)

# Factor persistence (AR(1))
phi = np.array([0.98, 0.95, 0.90])
sigma = np.array([0.0008, 0.0005, 0.0004])
factors = np.zeros((n, 3))
for t in range(1, n):
    factors[t] = phi * factors[t - 1] + sigma * np.random.randn(3)

# Loadings for Level, Slope, Curvature
loadings_true = np.vstack([
    np.ones(n_nodes),
    np.linspace(1, -1, n_nodes),
    (np.linspace(-1, 1, n_nodes) ** 2 - 0.33)
]).T

yields = 0.03 + factors @ loadings_true.T
yields_df = pd.DataFrame(yields, index=dates, columns=[f"{t}Y" for t in tenors])

# ======================================================
# 2. PCA ON YIELD CHANGES
# ======================================================
dy = yields_df.diff().dropna()
pca = PCA(n_components=3)
pca.fit(dy)
components = pca.components_
explained = pca.explained_variance_ratio_
eigvals = pca.explained_variance_

# ======================================================
# 3. PORTFOLIO OF 10 BONDS
# ======================================================
portfolio = pd.DataFrame({
    "Bond": [f"B{i+1}" for i in range(10)],
    "Maturity": np.linspace(0.5, 20, 10),
    "Coupon": np.linspace(0.015, 0.04, 10),
    "Notional": [1_000_000] * 10
})

def pv_bond(yield_curve, maturities, bond_mat, coupon, notional=1_000_000, freq=1):
    """Bond PV with basic discounting from zero curve."""
    times = np.arange(1, int(bond_mat * freq) + 1) / freq
    if len(times) == 0 or times[-1] < bond_mat:
        times = np.append(times, bond_mat)
    y_interp = np.interp(times, maturities, yield_curve)
    dfs = np.exp(-y_interp * times)
    cflow = np.repeat(coupon / freq * notional, len(times))
    cflow[-1] += notional
    return np.sum(cflow * dfs)

base_curve = yields_df.iloc[-1].values
vals = []
for _, row in portfolio.iterrows():
    vals.append(pv_bond(base_curve, tenors, row.Maturity, row.Coupon, row.Notional))
portfolio["MarketValue"] = vals

# ======================================================
# 4. KEY RATE DURATIONS
# ======================================================
bump = 0.0001
krd = np.zeros((10, n_nodes))
for i, row in portfolio.iterrows():
    base_pv = pv_bond(base_curve, tenors, row.Maturity, row.Coupon, row.Notional)
    for j in range(n_nodes):
        bumped = base_curve.copy()
        bumped[j] += bump
        bumped_pv = pv_bond(bumped, tenors, row.Maturity, row.Coupon, row.Notional)
        krd[i, j] = bumped_pv - base_pv
krd_df = pd.DataFrame(krd, index=portfolio.Bond, columns=yields_df.columns)

# ======================================================
# 5. MAP KRDs TO PCA LOADINGS
# ======================================================
pc_std = np.sqrt(eigvals)
exposures = np.zeros((10, 3))
for i in range(10):
    for k in range(3):
        exposures[i, k] = (krd[i, :] @ components[k] / bump) * pc_std[k]
exposures_df = pd.DataFrame(exposures, index=portfolio.Bond, columns=["PC1", "PC2", "PC3"])

# ======================================================
# 6. HISTORICAL BACKTEST: DAILY P&L, VAR, ES
# ======================================================
pvs = np.zeros((len(dy), 10))
for t in range(len(dy)):
    curve = yields_df.iloc[t + 1].values
    for i, row in portfolio.iterrows():
        pvs[t, i] = pv_bond(curve, tenors, row.Maturity, row.Coupon, row.Notional)

pnl = np.diff(pvs.sum(axis=1), prepend=pvs.sum(axis=1)[0])
VaR_99 = -np.percentile(pnl, 1)
ES_99 = -pnl[pnl <= np.percentile(pnl, 1)].mean()

# ======================================================
# 7. SCENARIO ANALYSIS (±1σ PC SHOCKS)
# ======================================================
scenarios = {}
for k in range(3):
    for sign, sname in zip([1, -1], ["+", "-"]):
        shock = sign * pc_std[k] * components[k]
        shocked_curve = base_curve + shock
        pnl_s = 0
        for _, row in portfolio.iterrows():
            pv_shock = pv_bond(shocked_curve, tenors, row.Maturity, row.Coupon, row.Notional)
            pv_base = pv_bond(base_curve, tenors, row.Maturity, row.Coupon, row.Notional)
            pnl_s += pv_shock - pv_base
        scenarios[f"{sname}1σ_PC{k+1}"] = pnl_s

# ======================================================
# 8. OUTPUT SUMMARY
# ======================================================
print("\n================ PCA & RISK SUMMARY ================\n")
print("Explained variance ratio:", np.round(explained, 4))
print("Portfolio Market Value: ", f"{portfolio.MarketValue.sum():,.2f}")
print("\nPortfolio PCA exposures ($ per 1σ PC move):")
print(exposures_df.sum().round(2))
print("\nScenario P&L (±1σ PCs):")
for k, v in scenarios.items():
    print(f"  {k}: {v:,.2f}")
print(f"\nHistorical VaR(99%): {VaR_99:,.2f}")
print(f"Historical ES(99%): {ES_99:,.2f}")

# ======================================================
# 9. PLOTS
# ======================================================
plt.figure(figsize=(7,3))
for k in range(3):
    plt.plot(tenors, components[k], marker='o', label=f"PC{k+1}")
plt.title("PCA Loadings by Maturity")
plt.xlabel("Maturity (Years)")
plt.ylabel("Loading")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(7,3))
plt.plot(pd.Series(pnl).cumsum(), label="Cumulative P&L")
plt.title("Simulated Historical Portfolio P&L")
plt.xlabel("Days"); plt.ylabel("Cumulative P&L ($)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
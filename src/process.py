import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
from typing import Tuple, Dict, Any
from pandas_datareader import data as pdr

# -----------------------------
# Utilities and configuration
# -----------------------------

np.random.seed(42) # reproducible

# Output folder for CSV/plots

OUT_DIR = "./pca_yield_study_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Default maturities (years) used as nodal points for the zero curve

DEFAULT_MATURITIES = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])

# -----------------------------
# 1) Yield curve simulation
# -----------------------------

def build_design_matrix(maturities: np.ndarray) -> np.ndarray:
    """
    Construct a simple 3-factor design matrix for zeros: level, slope, curvature.
    Columns are factors; rows are maturities.
    Normalizes each column to unit-length for interpretability.
    """
    L = np.ones_like(maturities) # level: same for all maturities
    S = 1.0 - (maturities / maturities.max()) # slope: decreasing with maturity
    peak = 3.0 # curvature peak around medium maturity (tuneable)
    C = np.exp(-0.5 * ((maturities - peak) / 2.0) ** 2) # hump shaped
    M = np.vstack([L, S, C]).T # shape (n_mats, 3)
    # Normalize each column (factor shape) to unit L2 norm
    M = M / np.linalg.norm(M, axis=0)
    return M # shape (n_mats, 3)

def simulate_factor_ar1(n_steps: int,
    phi: np.ndarray,
    sigma: np.ndarray,
    start: np.ndarray = None) -> np.ndarray:
    """
    Simulate a multivariate AR(1) for the factors.
    - n_steps: number of time steps
    - phi: (k,) autoregressive coefficients per factor (0<phi<1)
    - sigma: (k,) standard deviation of innovations (absolute yield units)
    - start: (k,) initial factor values
    Returns: factors array of shape (n_steps, k)
    """
    k = phi.shape[0]
    factors = np.zeros((n_steps, k))
    if start is None:
        start = np.zeros(k)
    factors[0] = start
    for t in range(1, n_steps):
        eps = np.random.normal(loc=0.0, scale=sigma, size=k)
        factors[t] = phi * factors[t - 1] + eps
    return factors

def construct_zero_curves(factors: np.ndarray,
    design: np.ndarray,
    baseline: float = 0.0) -> np.ndarray:
    """
    Build zero curves from factors and design matrix.
    zeros[t, j] = sum_k design[j,k] * factors[t,k] + baseline
    Input:
    - factors: (n_steps, k)
    - design: (n_mats, k)
    Output:
    - zeros: (n_steps, n_mats)
    """
    zeros = factors @ design.T # shape (n_steps, n_mats)
    zeros = zeros + baseline
    return zeros

# -----------------------------
# 2) Portfolio valuation helpers
# -----------------------------

def pv_bond(maturity: float,
    coupon_rate: float,
    notional: float,
    zero_curve_mats: np.ndarray,
    zero_curve_rates: np.ndarray,
    freq: int = 1) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Price a fixed-rate bullet bond using continuous discounting from zero curve nodes.
    - maturity: years until maturity
    - coupon_rate: annual coupon rate (decimal, e.g., 0.03)
    - notional: principal amount
    - zero_curve_mats: maturities for zero nodes (years)
    - zero_curve_rates: zero rates corresponding to nodes (decimal)
    - freq: coupon payments per year (1=annual, 2=semiannual, 4=quarterly)
    Returns:
    - pv: present value
    - times: cashflow times array
    - cf: cashflows array (cash at each time)
    - dfs: discount factors used (same length)
    Notes:
    - Uses linear interpolation for zero rates between nodes.
    - Defaults to continuous compounding for discount factors: DF = exp(-r * t).
    - Handles maturities that are not integer multiples of 1/freq by including final short stub.
    """
    if maturity <= 0:
        return 0.0, np.array([]), np.array([]), np.array([])

    step = 1.0 / freq
    if maturity <= step + 1e-12:
        times = np.array([maturity])
    else:
        # times from first coupon up to but excluding final maturity (if exact multiples),
        # then append maturity (ensuring final time equals maturity precisely).
        times = np.arange(step, maturity, step)
        times = np.append(times, maturity)

    # Build cashflows
    cf = np.zeros_like(times)
    for i, t in enumerate(times):
        if abs(t - maturity) < 1e-12:  # final payment includes principal
            prev = times[i - 1] if i > 0 else 0.0
            period_frac = t - prev
            # Coupon is proportional to the final accrual period length
            cf[i] = coupon_rate * notional * period_frac + notional
        else:
            cf[i] = coupon_rate * notional * step

    # Interpolate zero curve to get zero rate at each payment time
    interp = interp1d(zero_curve_mats, zero_curve_rates, kind="linear", fill_value="extrapolate", assume_sorted=True)
    zero_at_t = interp(times)
    dfs = np.exp(-zero_at_t * times)  # continuous compounding
    pv = (cf * dfs).sum()
    return float(pv), times, cf, dfs

def compute_krd_for_bond(maturity: float,
    coupon_rate: float,
    notional: float,
    zero_curve_mats: np.ndarray,
    base_zero_rates: np.ndarray,
    bump: float = 0.0001,
    freq: int = 1) -> np.ndarray:
    """
    Compute Key-Rate Durations (in dollar PV change) for each node of zero_curve_mats
    for a single bond. KRD[j] = PV(bumped at node j by +bump) - PV(base).
    Returns a vector of length n_mats (dollar change for +1bp by default).
    """
    pv_base, _, _, _ = pv_bond(maturity, coupon_rate, notional, zero_curve_mats, base_zero_rates, freq=freq)
    n_mats = len(zero_curve_mats)
    krd = np.zeros(n_mats)
    for j in range(n_mats):
        bumped = base_zero_rates.copy()
        bumped[j] = bumped[j] + bump
        pv_bumped, _, _, _ = pv_bond(maturity, coupon_rate, notional, zero_curve_mats, bumped, freq=freq)
        krd[j] = pv_bumped - pv_base  # dollar change for +bump
    return krd

# -----------------------------
# 3) PCA analysis
# -----------------------------

def run_pca_on_dy(yields_df: pd.DataFrame, n_components: int = 3) -> Dict[str, Any]:
    """
    Run PCA on daily yield changes (dy = diff of yields).
    Inputs:
    - yields_df: DataFrame with columns representing nodal maturities (in order),
    rows are dates, values are zero rates (decimal).
    Returns:
    - dict with keys: 'pca' (sklearn PCA), 'components' (array n_components x n_mats),
    'explained_variance_ratio', 'explained_variance', 'scores' (time x n_components),
    'dy' (the used dy DataFrame).
    """
    # compute daily changes (dy)
    dy = yields_df.diff().dropna()
    # PCA on rows of dy (observations = days, features = nodal yields)
    pca = PCA(n_components=n_components)
    pca.fit(dy.values)
    components = pca.components_ # (n_components, n_mats)
    explained_variance_ratio = pca.explained_variance_ratio_
    explained_variance = pca.explained_variance_
    # compute scores (projection of dy onto PC directions)
    scores = dy.values @ components.T # shape (n_obs, n_components)
    return {
    "pca": pca,
    "components": components,
    "explained_variance_ratio": explained_variance_ratio,
    "explained_variance": explained_variance,
    "scores": scores,
    "dy": dy
    }
    
# -----------------------------
# 4) Mapping KRDs to PCs & risk contributions
# -----------------------------

def map_krd_to_pc_exposure(krd_matrix: np.ndarray,
    pc_components: np.ndarray,
    pc_std: np.ndarray,
    bump_unit: float = 0.0001) -> np.ndarray:
    """
    Map bond-level KRD matrix to exposures to PCs.
    Inputs:
    - krd_matrix: shape (n_bonds, n_mats) dollar PV change for +bump_unit at each node
    - pc_components: (n_components, n_mats) PCA loading vectors (per unit yield change)
    - pc_std: (n_components,) standard deviation of PC scores (in yield units)
    - bump_unit: the unit used for KRDs (default 1bp = 0.0001)
    Output:
    - exposures: (n_bonds, n_components) dollar P&L per 1 *pc_std move in that PC
    Explanation:
    If a PC k causes a yield change vector delta_yields = pc_std[k] * pc_components[k],
    and KRD gives dollar change per +1bp at each node, then dollar P&L = sum_j KRD_j * (delta_yields_j / bump_unit).
    """
    n_bonds = krd_matrix.shape[0]
    n_components = pc_components.shape[0]
    exposures = np.zeros((n_bonds, n_components))
    for i in range(n_bonds):
        for k in range(n_components):
            # yield change vector caused by a 1 * pc_std move in PC k
            delta_yields = pc_std[k] * pc_components[k]  # shape (n_mats,)
            # convert node delta_yields to number of bump_units, then multiply by dollar/KRD
            multiplier = (delta_yields / bump_unit)
            exposures[i, k] = (krd_matrix[i, :] * multiplier).sum()
    return exposures

def portfolio_risk_contributions(portfolio_pc_sens_total: np.ndarray,
    pc_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate approximate risk contributions from PCs using variance of scores and sensitivities.
    Inputs:
    - portfolio_pc_sens_total: (n_components,) total portfolio sensitivity per PC (dollar per 1*pc_std)
    - pc_scores: (n_obs, n_components) historical PC scores (units consistent with pc_std)
    Returns:
    - contrib (n_components,) = contribution to portfolio variance (in squared dollar units)
    - contrib_pct (n_components,) = fraction of variance explained per PC
    Formula:
    If portfolio P&L from PC k is S_k * score_k, then var_total = sum_k S_k^2 * var(score_k) assuming uncorrelated PCs.
    Note:
    If scores are correlated (they should not be for PCA but due to sample noise might be), this is an approximation.
    """
    var_scores = np.var(pc_scores, axis=0, ddof=1)
    contrib = (portfolio_pc_sens_total ** 2) * var_scores
    total = contrib.sum()
    contrib_pct = contrib / total if total != 0 else np.zeros_like(contrib)
    return contrib, contrib_pct

# -----------------------------
# 5) Scenario analysis using PCs
# -----------------------------

def scenario_pc_shock(base_zero_rates: np.ndarray,
    pc_components: np.ndarray,
    pc_std: np.ndarray,
    shock_component: int,
    shock_sign: int,
    maturities_nodes: np.ndarray,
    portfolio_df: pd.DataFrame,
    freq: int = 1) -> Dict[str, Any]:
    """
    Apply a ±1σ shock on one PC and compute re-priced portfolio P&L.
    shock_sign: +1 or -1
    shock_component: index of PC (0-based)
    Returns a dict with per-bond P&L and total P&L and shocked zero curve.
    """
    # reconstruct yield perturbation vector
    shock = shock_sign * pc_std[shock_component] * pc_components[shock_component]
    shocked_curve = base_zero_rates + shock
    # reprice portfolio
    per_bond_pv_shocked = []
    for _, row in portfolio_df.iterrows():
        pv_s, _, _, _ = pv_bond(row['maturity'], row['coupon'], row['notional'],
                                 maturities_nodes, shocked_curve, freq=freq)
        per_bond_pv_shocked.append(pv_s)
    per_bond_pv_shocked = np.array(per_bond_pv_shocked)
    pv_base = np.array(portfolio_df['market_value'].values)
    pnl = per_bond_pv_shocked - pv_base
    return {
        "shock_name": f"{('+' if shock_sign>0 else '-') }1sigma_PC{shock_component+1}",
        "shocked_curve": shocked_curve,
        "per_bond_pv_shocked": per_bond_pv_shocked,
        "per_bond_pnl": pnl,
        "total_pnl": float(pnl.sum())
    }
    
# -----------------------------
# 6) Backtest and VaR
# -----------------------------

def historical_backtest(yields_df: pd.DataFrame,
    portfolio_df: pd.DataFrame,
    maturities_nodes: np.ndarray,
    freq: int = 1) -> Dict[str, Any]:
    """
    Re-price the portfolio for each date in yields_df and compute daily P&L series.
    Returns:
    - pv_time: array (n_dates, n_bonds)
    - pnl_time: array (n_dates-1, n_bonds)
    - pnl_total_time: array (n_dates-1,)
    - VaR_99, ES_99 (historical simulation)
    """
    dates = yields_df.index
    n_dates = len(dates)
    n_bonds = len(portfolio_df)
    pv_time = np.zeros((n_dates, n_bonds))
    for t in range(n_dates):
        curve = yields_df.iloc[t].values
        for i, row in portfolio_df.iterrows():
            pv_t, _, _, _ = pv_bond(row['maturity'], row['coupon'], row['notional'],
                                    maturities_nodes, curve, freq=freq)
            pv_time[t, i] = pv_t
    pnl_time = pv_time[1:, :] - pv_time[:-1, :]  # daily P&L per bond
    pnl_total_time = pnl_time.sum(axis=1)
    # Historical-simulation VaR at 99% (one-day)
    var_99 = -np.percentile(pnl_total_time, 1)  # loss positive
    tail_cutoff = np.percentile(pnl_total_time, 1)
    es_99 = -pnl_total_time[pnl_total_time <= tail_cutoff].mean() if (pnl_total_time <= tail_cutoff).any() else 0.0

    return {
        "pv_time": pv_time,
        "pnl_time": pnl_time,
        "pnl_total_time": pnl_total_time,
        "VaR_99": float(var_99),
        "ES_99": float(es_99)
    }
    
# -----------------------------
# 7) End-to-end pipeline function
# -----------------------------

def run_full_pipeline(
    n_steps: int = 1000,
    maturities_nodes: np.ndarray = DEFAULT_MATURITIES,
    phi: np.ndarray = np.array([0.98, 0.9, 0.85]),
    sigma_f: np.ndarray = np.array([0.01, 0.005, 0.003]),
    factor_start: np.ndarray = None,
    baseline_rate: float = 0.01,
    portfolio_spec: pd.DataFrame = None,
    bump_unit: float = 0.0001,
    n_pcs: int = 3,
    coupon_freq: int = 1
    ) -> Dict[str, Any]:
    """
    Execute the entire workflow (simulate, PCA, KRD, mapping, scenarios, backtest).
    Returns a dictionary with many outputs (see end of function).
    """
    # 1) Build design matrix
    design = build_design_matrix(maturities_nodes) # shape (n_mats, 3)
    # 2) Simulate factors AR(1) -> zero curves
    k = design.shape[1]
    if factor_start is None:
        factor_start = np.zeros(k)
        factor_start[0] = 0.02  # small positive initial level (tweakable)
        factor_start[1] = -0.01
        factor_start[2] = 0.005
    factors = simulate_factor_ar1(n_steps=n_steps, phi=phi, sigma=sigma_f, start=factor_start)
    zeros = construct_zero_curves(factors, design, baseline=baseline_rate)  # shape (n_steps, n_mats)

    # Dates: business days ending today
    dates = pd.date_range(end=pd.Timestamp.today(), periods=n_steps, freq="B")
    col_names = [f"{m}Y" for m in maturities_nodes]
    yields_df = pd.DataFrame(zeros, index=dates, columns=col_names)

    # 3) PCA on yield changes
    pca_results = run_pca_on_dy(yields_df, n_components=n_pcs)
    pcs = pca_results["components"]          # (n_pcs, n_mats)
    explained_variance_ratio = pca_results["explained_variance_ratio"]
    explained_variance = pca_results["explained_variance"]
    scores = pca_results["scores"]
    dy = pca_results["dy"]

    # 4) Build portfolio (if not provided, create example portfolio of 10 bonds)
    if portfolio_spec is None:
        portfolio_spec = pd.DataFrame({
            'bond_id': [f'B{i+1}' for i in range(10)],
            'maturity': [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 20],
            'coupon': [0.01, 0.015, 0.02, 0.01, 0.025, 0.03, 0.02, 0.025, 0.03, 0.035],
            'notional': [1_000_000] * 10
        })
    portfolio = portfolio_spec.copy().reset_index(drop=True)

    # 5) Compute market values at the latest date and KRDs
    latest_zero = yields_df.iloc[-1].values  # base curve
    n_bonds = len(portfolio)
    n_mats = len(maturities_nodes)
    krd_matrix = np.zeros((n_bonds, n_mats))
    market_values = np.zeros(n_bonds)
    for i, row in portfolio.iterrows():
        pv, _, _, _ = pv_bond(row['maturity'], row['coupon'], row['notional'],
                            maturities_nodes, latest_zero, freq=coupon_freq)
        market_values[i] = pv
        krd_matrix[i, :] = compute_krd_for_bond(row['maturity'], row['coupon'], row['notional'],
                                            maturities_nodes, latest_zero, bump=bump_unit, freq=coupon_freq)

    portfolio['market_value'] = market_values

    # 6) Map KRDs to PC exposures
    pc_std = np.sqrt(explained_variance)  # standard deviation of PC scores
    exposures = map_krd_to_pc_exposure(krd_matrix, pcs, pc_std, bump_unit=bump_unit)  # shape (n_bonds, n_pcs)
    exposures_df = pd.DataFrame(exposures, index=portfolio['bond_id'], columns=[f"PC{i+1}" for i in range(n_pcs)])

    # Aggregate portfolio-level sensitivities (sum over bonds)
    portfolio_pc_total = exposures_df.sum(axis=0).values  # (n_pcs,)

    # 7) Risk contributions
    contrib, contrib_pct = portfolio_risk_contributions(portfolio_pc_total, scores)

    # 8) Scenario analysis ±1σ per PC
    scenarios = []
    for k_idx in range(n_pcs):
        for sign in [+1, -1]:
            s = scenario_pc_shock(latest_zero, pcs, pc_std, shock_component=k_idx,
                                shock_sign=sign, maturities_nodes=maturities_nodes,
                                portfolio_df=portfolio, freq=coupon_freq)
            scenarios.append(s)

    # 9) Historical backtest (re-price for all dates)
    backtest_results = historical_backtest(yields_df, portfolio, maturities_nodes, freq=coupon_freq)

    # Optional: Compute daily P&L approximated using PC projection (linear delta)
    # Compute daily PC scores (already available) and portfolio linear P&L via portfolio_pc_total
    pc_scores = scores  # shape (n_obs, n_pcs)
    # For daily P&L approximated: we must ensure units align; exposures are dollar P&L per 1*pc_std move.
    # pc_scores are in units of yield-change corresponding to 1 unit of component, so to map to "units of pc_std"
    # we can compute scaled_scores = pc_scores / pc_std[None, :]  (number of 'sigma' moves).
    # Then approximate daily P&L = sum_k portfolio_pc_total[k] * (pc_scores[:,k] / pc_std[k])
    scaled_scores = pc_scores / pc_std[np.newaxis, :]
    approx_daily_pl = (scaled_scores * portfolio_pc_total[np.newaxis, :]).sum(axis=1)  # len = n_obs

    # Build final results dictionary with many items for inspection / saving
    outputs = {
        "yields_df": yields_df,
        "dy": dy,
        "pca": pca_results["pca"],
        "pc_components": pcs,
        "explained_variance_ratio": explained_variance_ratio,
        "explained_variance": explained_variance,
        "pc_std": pc_std,
        "pc_scores": pc_scores,
        "portfolio": portfolio,
        "krd_matrix": krd_matrix,
        "krd_df": pd.DataFrame(krd_matrix, index=portfolio['bond_id'], columns=col_names),
        "exposures_df": exposures_df,
        "portfolio_pc_total": pd.Series(portfolio_pc_total, index=[f"PC{i+1}" for i in range(n_pcs)]),
        "risk_contrib": contrib,
        "risk_contrib_pct": contrib_pct,
        "scenarios": scenarios,
        "backtest": backtest_results,
        "approx_daily_pl_from_pcs": approx_daily_pl,
        "dates_for_scores": dy.index  # these correspond to pc_scores rows
    }

    # 10) Save key outputs to CSVs for convenience
    outputs["krd_df"].to_csv(os.path.join(OUT_DIR, "krd_matrix.csv"))
    outputs["exposures_df"].to_csv(os.path.join(OUT_DIR, "pc_exposures_per_bond.csv"))
    pd.DataFrame(pcs.T, index=col_names, columns=[f"PC{i+1}" for i in range(n_pcs)]).to_csv(os.path.join(OUT_DIR, "pc_loadings.csv"))
    pd.DataFrame(outputs["explained_variance_ratio"].reshape(1, -1), columns=[f"PC{i+1}_expl_var_ratio" for i in range(n_pcs)]).to_csv(os.path.join(OUT_DIR, "pc_explained_variance_ratio.csv"), index=False)
    scen_df = pd.DataFrame([{"scenario": s["shock_name"], "total_pnl": s["total_pnl"]} for s in scenarios]).set_index("scenario")
    scen_df.to_csv(os.path.join(OUT_DIR, "scenario_pnl.csv"))
    # Save backtest P&L time series
    pd.DataFrame(outputs["backtest"]["pnl_total_time"], index=dates[1:], columns=["pnl_total"]).to_csv(os.path.join(OUT_DIR, "backtest_pnl_total_time.csv"))

    return outputs

# -----------------------------
# 8) Plotting helpers (optional visualizations)
# -----------------------------

def plot_pc_loadings(maturities_nodes: np.ndarray, pc_components: np.ndarray, out_file: str = None):
    """Plot PCA loadings across maturities."""
    plt.figure(figsize=(10, 5))
    n_pcs = pc_components.shape[0]
    for i in range(n_pcs):
        plt.plot(maturities_nodes, pc_components[i, :], marker='o', label=f"PC{i+1} loading")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Loading (per unit yield change)")
    plt.title("PCA Loadings (components) across maturities")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    if out_file:
        plt.savefig(out_file, bbox_inches="tight")
    plt.show()

def plot_cumulative_pnl(dates: pd.DatetimeIndex, pnl_series: np.ndarray, out_file: str = None):
    """Plot cumulative P&L time series."""
    cum = np.cumsum(pnl_series)
    plt.figure(figsize=(10, 5))
    plt.plot(dates, cum, linewidth=1.0)
    plt.title("Cumulative portfolio P&L")
    plt.xlabel("Date")
    plt.ylabel("Cumulative P&L")
    plt.grid(True, linestyle="--", alpha=0.5)
    if out_file:
        plt.savefig(out_file, bbox_inches="tight")
    plt.show()
    
# -----------------------------
# 9) Run script when executed directly
# -----------------------------

if __name__ == "__main__":
    # Configuration - you can tweak these parameters before running
    N_STEPS = 1000
    MATURITIES = DEFAULT_MATURITIES.copy()
    PHI = np.array([0.98, 0.9, 0.85])
    SIGMA_F = np.array([0.01, 0.005, 0.003]) # daily vol of factor shocks (in absolute yield units)
    BASELINE_RATE = 0.01 # 1% baseline zero rate
    BUMP_BP = 0.0001 # 1bp bump for KRD
    N_PCS = 3
    COUPON_FREQ = 1 # annual coupons
    
    # Example portfolio (same as in the previous run). You may replace this DataFrame as desired.
    example_portfolio = pd.DataFrame({
        'bond_id': [f'B{i+1}' for i in range(10)],
        'maturity': [0.5, 1, 1.5, 2, 3, 4, 5, 7, 10, 20],
        'coupon': [0.01, 0.015, 0.02, 0.01, 0.025, 0.03, 0.02, 0.025, 0.03, 0.035],
        'notional': [1_000_000] * 10
    })

    print("Running full pipeline (simulated yields). This may take a few seconds...")
    results = run_full_pipeline(n_steps=N_STEPS,
                                maturities_nodes=MATURITIES,
                                phi=PHI,
                                sigma_f=SIGMA_F,
                                baseline_rate=BASELINE_RATE,
                                portfolio_spec=example_portfolio,
                                bump_unit=BUMP_BP,
                                n_pcs=N_PCS,
                                coupon_freq=COUPON_FREQ)

    # Print summary information
    print("\n--- PCA explained variance ratio ---")
    for i, v in enumerate(results["explained_variance_ratio"], 1):
        print(f"PC{i}: {v:.4f}")
    print("\nPortfolio market value (sum):", f"{results['portfolio']['market_value'].sum():,.2f}")
    print("\nPortfolio sensitivity to PCs (total across bonds):")
    print(results["portfolio_pc_total"].round(2))
    print("\nRisk contribution percentages from PCs (approx):")
    for i, pct in enumerate(results["risk_contrib_pct"], 1):
        print(f"  PC{i}: {pct*100:.1f}%")

    print("\nBacktest VaR(99%):", f"{results['backtest']['VaR_99']:,.2f}")
    print("Backtest ES(99%):", f"{results['backtest']['ES_99']:,.2f}")

    # Produce plots and save
    pc_loadings_plot = os.path.join(OUT_DIR, "pc_loadings.png")
    plot_pc_loadings(MATURITIES, results["pc_components"], out_file=pc_loadings_plot)
    backtest_plot = os.path.join(OUT_DIR, "cumulative_pnl.png")
    # dates for P&L are yields_df.index[1:]
    plot_cumulative_pnl(results["yields_df"].index[1:], results["backtest"]["pnl_total_time"], out_file=backtest_plot)

    print(f"\nAll key CSV outputs and plots saved to '{OUT_DIR}'.")
    print("Files created (examples):")
    print(" - krd_matrix.csv")
    print(" - pc_exposures_per_bond.csv")
    print(" - pc_loadings.csv")
    print(" - scenario_pnl.csv")
    print(" - backtest_pnl_total_time.csv")
    print("\nTo use real historical data, replace 'yields_df' inside run_full_pipeline with your DataFrame of zero curves.")
    print("If you'd like, I can add a helper to fetch specific Treasury maturities from FRED and bootstrap/interpolate a zero curve; ask and I will provide that addition.")

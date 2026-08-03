from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares

import paleochannel_phase_validation_advanced_v5 as mod   # Issue 5 fix

C0 = mod.C0


# ---------------------------------------------------------------------------
# LEGACY PATH (deprecated) — analytic (eps_real, eps_imag)-from-phase retrieval
# Retained for reproducibility of the v5 study; NOT for permittivity retrieval.
# See the CORRECTION section in the module docstring.
# ---------------------------------------------------------------------------

def forward_channel_response(
    eps_channel_complex,
    channel_depth_m,
    cfg,
    theta_deg_array,
    membership=1.0,
):
 
    theta_deg_array = np.atleast_1d(np.asarray(theta_deg_array, dtype=float))
    theta = np.deg2rad(theta_deg_array)
    n = theta.shape[0]

    M = np.broadcast_to(np.asarray(membership, dtype=float), (n,)).copy()

    # Surface dielectric
    if cfg.dielectric_mode == "constant":
        eps_surface = (cfg.background_epsilon_complex
                       + (cfg.channel_epsilon_complex - cfg.background_epsilon_complex) * M)
    else:
        moisture = cfg.background_moisture + (cfg.channel_moisture - cfg.background_moisture) * M
        eps_surface = mod.dielectric_from_moisture(moisture, cfg)

    # Surface roughness
    s = cfg.background_roughness_rms_m + (cfg.channel_roughness_rms_m - cfg.background_roughness_rms_m) * M
    l = np.full(n, float(cfg.correlation_length_m))

    surface = mod.iem_inspired_surface_fields(
        eps_surface, theta, s, l, cfg.freq_hz, spectrum=cfg.roughness_spectrum)
    if cfg.mode == "surface":
        E_surface_hh, E_surface_vv = surface["R_hh"], surface["R_vv"]
    else:
        E_surface_hh, E_surface_vv = surface["E_surface_hh"], surface["E_surface_vv"]

    E_volume_hh = np.zeros(n, dtype=np.complex128)
    E_volume_vv = np.zeros(n, dtype=np.complex128)

    for i, layer in enumerate(cfg.layers):
        if i == 0:
            eps_channel = eps_channel_complex
            layer_own_depth = channel_depth_m
        else:
            eps_channel = layer.epsilon_channel
            layer_own_depth = layer.depth_m

        eps_layer = layer.epsilon_background + (eps_channel - layer.epsilon_background) * M

        # Total depth to reflecting interface (Issue 4 fix)
        surface_depth = cfg.background_depth_m + cfg.channel_depth_amplitude_m * M
        if layer.depth_follows_channel:
            total_layer_depth = surface_depth + layer_own_depth * M
        else:
            total_layer_depth = np.full(n, float(layer_own_depth))

        amp = layer.amplitude * (M if layer.scale_by_membership else np.ones(n))
        lf = mod._layer_field(eps_surface, eps_layer, theta, cfg.freq_hz, total_layer_depth, amp)
        E_volume_hh += lf["E_hh"]
        E_volume_vv += lf["E_vv"]

    E_total_hh = E_surface_hh + E_volume_hh
    E_total_vv = E_surface_vv + E_volume_vv

    sigma0_hh_db = 10.0 * np.log10(np.maximum(np.abs(E_total_hh) ** 2, 1e-12))
    sigma0_vv_db = 10.0 * np.log10(np.maximum(np.abs(E_total_vv) ** 2, 1e-12))
    phi_hhvv = mod._wrapped_phase_difference(
        np.angle(E_total_hh), np.angle(E_total_vv), cfg.phase_shift_hhvv_rad)

    wavelength = C0 / cfg.freq_hz
    kz_surf = ((4.0 * np.pi / wavelength)
               * cfg.b_perp_m / (cfg.slant_range_m * np.maximum(np.sin(theta), 1e-8)))
    eps_real = max(float(np.real(eps_channel_complex)), 1.000001)
    kz_vol = kz_surf * (eps_real * np.cos(theta)) / np.sqrt(
        np.maximum(eps_real - np.sin(theta) ** 2, 1e-8))

    return {
        "sigma0_hh_db": sigma0_hh_db,
        "sigma0_vv_db": sigma0_vv_db,
        "phi_hhvv_rad": phi_hhvv,
        "kz_vol": kz_vol,
    }


# ---------------------------------------------------------------------------
# Least-squares inversion
# ---------------------------------------------------------------------------

def _residual_vector(
    p, estimate_depth, fixed_depth,
    theta_deg_obs, vv_obs, hh_obs, phi_obs,
    cfg, phase_weight_deg_equiv,
):
    eps_r, eps_i = p[0], p[1]
    depth = p[2] if estimate_depth else fixed_depth
    eps_complex = eps_r - 1j * eps_i
    pred = forward_channel_response(eps_complex, depth, cfg, theta_deg_obs, membership=1.0)

    r_vv = pred["sigma0_vv_db"] - vv_obs
    r_hh = pred["sigma0_hh_db"] - hh_obs
    r_phi_rad = np.angle(np.exp(1j * (pred["phi_hhvv_rad"] - phi_obs)))
    r_phi = np.rad2deg(r_phi_rad) * float(phase_weight_deg_equiv)

    return np.concatenate([r_vv, r_hh, r_phi])


def estimate_channel_permittivity(
    theta_deg_obs,
    sigma0_vv_db_obs,
    sigma0_hh_db_obs,
    phi_hhvv_rad_obs,
    cfg,
    channel_depth_m=None,
    eps_real_bounds=(1.5, 30.0),
    eps_imag_bounds=(0.0, 10.0),
    depth_bounds=(0.05, 10.0),
    n_restarts=20,
    rng_seed=0,
    phase_weight_deg_equiv=1.0,
):

    theta_deg_obs = np.atleast_1d(np.asarray(theta_deg_obs, dtype=float))
    sigma0_vv_db_obs = np.atleast_1d(np.asarray(sigma0_vv_db_obs, dtype=float))
    sigma0_hh_db_obs = np.atleast_1d(np.asarray(sigma0_hh_db_obs, dtype=float))
    phi_hhvv_rad_obs = np.atleast_1d(np.asarray(phi_hhvv_rad_obs, dtype=float))

    warnings.warn(
        "estimate_channel_permittivity is deprecated: it fits (eps_real, eps_imag) "
        "from co-pol phase, but eps_imag is unidentifiable from co-pol sigma0 at "
        "P-band. Use estimate_surface_dielectric_roughness (LUT, linear, "
        "(eps_real, ks)) instead.",
        DeprecationWarning, stacklevel=2)

    estimate_depth = channel_depth_m is None
    rng = np.random.default_rng(rng_seed)

    lb = [eps_real_bounds[0], eps_imag_bounds[0]] + ([depth_bounds[0]] if estimate_depth else [])
    ub = [eps_real_bounds[1], eps_imag_bounds[1]] + ([depth_bounds[1]] if estimate_depth else [])

    best_fit = None
    restarts = []

    for _ in range(n_restarts):
        eps_r0 = rng.uniform(*eps_real_bounds)
        eps_i0 = rng.uniform(*eps_imag_bounds)
        p0 = [eps_r0, eps_i0] + ([rng.uniform(*depth_bounds)] if estimate_depth else [])

        fit = least_squares(
            _residual_vector, p0, bounds=(lb, ub),
            args=(estimate_depth, channel_depth_m,
                  theta_deg_obs, sigma0_vv_db_obs, sigma0_hh_db_obs, phi_hhvv_rad_obs,
                  cfg, phase_weight_deg_equiv),
        )

        eps_r_fit, eps_i_fit = fit.x[0], fit.x[1]
        depth_fit = fit.x[2] if estimate_depth else channel_depth_m
        restarts.append({
            "p0": p0,
            "eps_real": eps_r_fit,
            "eps_imag": eps_i_fit,
            "depth_m": depth_fit,
            "cost": fit.cost,
        })

        if best_fit is None or fit.cost < best_fit.cost:
            best_fit = fit

    eps_r_fit, eps_i_fit = best_fit.x[0], best_fit.x[1]
    depth_fit = best_fit.x[2] if estimate_depth else channel_depth_m

    try:
        n_obs = len(best_fit.fun)
        n_params = len(best_fit.x)
        dof = max(n_obs - n_params, 1)
        resid_var = float(np.sum(best_fit.fun ** 2) / dof)
        JTJ = best_fit.jac.T @ best_fit.jac
        cov = resid_var * np.linalg.pinv(JTJ)
        param_std = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        param_std = np.full(len(best_fit.x), np.nan)

    n_distinct = len({
        (round(r["eps_real"], 1), round(r["eps_imag"], 2))
        for r in restarts
    })

    return {
        "eps_real": eps_r_fit,
        "eps_imag": eps_i_fit,
        "epsilon_complex": eps_r_fit - 1j * eps_i_fit,
        "depth_m": depth_fit,
        "eps_real_std": param_std[0],
        "eps_imag_std": param_std[1],
        "depth_std": param_std[2] if estimate_depth else 0.0,
        "success": best_fit.success,
        "cost": best_fit.cost,
        "residual_rmse": float(np.sqrt(np.mean(best_fit.fun ** 2))),
        "n_observations": len(theta_deg_obs),
        "n_restarts": n_restarts,
        "n_distinct_local_minima": n_distinct,
        "restarts": restarts,
        "note": (
            f"{n_distinct} distinct local minima found across {n_restarts} restarts — "
            "if this is more than 1–2, treat the point estimate with caution and inspect "
            "the cost landscape (see identifiability_scan) before trusting it."
        ),
    }


# ---------------------------------------------------------------------------
# Monte Carlo: inversion uncertainty from measurement noise
# ---------------------------------------------------------------------------

def mc_amplitude_phase_to_dielectric(
    theta_deg_obs,
    sigma0_vv_db_true,
    sigma0_hh_db_true,
    phi_hhvv_rad_true,
    cfg,
    channel_depth_m=None,
    n_realizations=200,
    amp_noise_std=None,
    phase_noise_std_rad=None,
    n_restarts=5,
    rng_seed=0,
    phase_weight_deg_equiv=1.0,
):
  
    warnings.warn(
        "mc_amplitude_phase_to_dielectric is deprecated: use "
        "mc_copol_to_dielectric_roughness (LUT-based, linear sigma0).",
        DeprecationWarning, stacklevel=2)

    rng = np.random.default_rng(rng_seed)

    theta_deg_obs = np.atleast_1d(np.asarray(theta_deg_obs, dtype=float))
    sigma0_vv_db_true = np.atleast_1d(np.asarray(sigma0_vv_db_true, dtype=float))
    sigma0_hh_db_true = np.atleast_1d(np.asarray(sigma0_hh_db_true, dtype=float))
    phi_hhvv_rad_true = np.atleast_1d(np.asarray(phi_hhvv_rad_true, dtype=float))

    if amp_noise_std is None:
        amp_noise_std = 8.686 * cfg.amp_noise_std_channel
    if phase_noise_std_rad is None:
        phase_noise_std_rad = cfg.phase_noise_std_channel_rad

    eps_real_samples, eps_imag_samples, depth_samples = [], [], []
    failed = 0

    for _ in range(n_realizations):
        vv_noisy = sigma0_vv_db_true + rng.normal(0.0, amp_noise_std, len(sigma0_vv_db_true))
        hh_noisy = sigma0_hh_db_true + rng.normal(0.0, amp_noise_std, len(sigma0_hh_db_true))
        phi_noisy = np.angle(np.exp(
            1j * (phi_hhvv_rad_true
                  + rng.normal(0.0, phase_noise_std_rad, len(phi_hhvv_rad_true)))))

        try:
            r = estimate_channel_permittivity(
                theta_deg_obs, vv_noisy, hh_noisy, phi_noisy, cfg,
                channel_depth_m=channel_depth_m,
                n_restarts=n_restarts,
                rng_seed=int(rng.integers(0, 2 ** 31)),
                phase_weight_deg_equiv=phase_weight_deg_equiv,
            )
            eps_real_samples.append(r["eps_real"])
            eps_imag_samples.append(r["eps_imag"])
            depth_samples.append(r["depth_m"])
        except Exception:
            failed += 1

    def _s(arr):
        a = np.asarray(arr)
        return dict(
            mean=float(np.nanmean(a)), std=float(np.nanstd(a)),
            p05=float(np.nanpercentile(a, 5)), p95=float(np.nanpercentile(a, 95)),
            samples=a,
        )

    return dict(
        eps_real=_s(eps_real_samples),
        eps_imag=_s(eps_imag_samples),
        depth_m=_s(depth_samples),
        n_realizations=n_realizations,
        failed=failed,
        noise_params=dict(
            amp_noise_db=amp_noise_std,
            phase_noise_deg=np.rad2deg(phase_noise_std_rad),
        ),
    )


# ---------------------------------------------------------------------------
# Monte Carlo: coherence → reflector depth uncertainty
# ---------------------------------------------------------------------------

def mc_coherence_to_depth(
    gamma_obs_true,
    kz_vol,
    cfg,
    n_realizations=200,
    n_looks=None,
    gamma_temp=None,
    rng_seed=0,
):
 
    gamma_obs_true = np.asarray(gamma_obs_true, dtype=float)
    kz_vol = np.asarray(kz_vol, dtype=float)

    if n_looks is None:
        n_looks = int(cfg.coherence_window_px) ** 2

    snr_lin = 10.0 ** (cfg.snr_db / 10.0)
    gamma_snr = snr_lin / (1.0 + snr_lin)

    if gamma_temp is None:
        gamma_temp = np.full_like(gamma_obs_true, cfg.temporal_coherence_background)

    rng = np.random.default_rng(rng_seed)
    sigma_gamma = (1.0 - gamma_obs_true ** 2) / np.sqrt(2.0 * n_looks)

    depth_stack = []
    for _ in range(n_realizations):
        gamma_noisy = np.clip(
            gamma_obs_true + rng.normal(0.0, sigma_gamma), 1e-6, 1.0 - 1e-6)
        gamma_vol_est = np.clip(
            gamma_noisy / (np.maximum(gamma_temp, 1e-6) * gamma_snr), 1e-6, 1.0 - 1e-6)
        depth = -np.log(gamma_vol_est) / (
            cfg.volume_decorrelation_strength * np.abs(kz_vol) + 1e-10)
        depth = np.clip(depth, 0.0, None)
        depth_stack.append(depth)

    depth_stack = np.array(depth_stack)

    # Nominal depth from the true coherence (= reflector_depth_m)
    gamma_vol_nominal = np.clip(
        gamma_obs_true / (np.maximum(gamma_temp, 1e-6) * gamma_snr), 1e-6, 1.0 - 1e-6)
    nominal_depth = -np.log(gamma_vol_nominal) / (
        cfg.volume_decorrelation_strength * np.abs(kz_vol) + 1e-10)

    return dict(
        depth_mean=np.nanmean(depth_stack, axis=0),
        depth_std=np.nanstd(depth_stack, axis=0),
        depth_p05=np.nanpercentile(depth_stack, 5, axis=0),
        depth_p95=np.nanpercentile(depth_stack, 95, axis=0),
        n_realizations=n_realizations,
        n_looks=n_looks,
        sigma_gamma=sigma_gamma,
        nominal_depth=nominal_depth,   # = reflector_depth_m (Issue 4)
    )


# ---------------------------------------------------------------------------
# Identifiability scan
# ---------------------------------------------------------------------------

def identifiability_scan(
    theta_deg_obs,
    sigma0_vv_db_obs,
    sigma0_hh_db_obs,
    phi_hhvv_rad_obs,
    cfg,
    channel_depth_m,
    eps_real_grid=None,
    eps_imag_grid=None,
    phase_weight_deg_equiv=1.0,
):
    warnings.warn(
        "identifiability_scan (over eps_real, eps_imag) is deprecated: use "
        "identifiability_scan_lut (over eps_real, ks) for the real method.",
        DeprecationWarning, stacklevel=2)

    if eps_real_grid is None:
        eps_real_grid = np.linspace(1.5, 25.0, 60)
    if eps_imag_grid is None:
        eps_imag_grid = np.linspace(0.0, 5.0, 60)

    theta_deg_obs = np.atleast_1d(theta_deg_obs)
    sigma0_vv_db_obs = np.atleast_1d(sigma0_vv_db_obs)
    sigma0_hh_db_obs = np.atleast_1d(sigma0_hh_db_obs)
    phi_hhvv_rad_obs = np.atleast_1d(phi_hhvv_rad_obs)

    rmse_grid = np.zeros((len(eps_imag_grid), len(eps_real_grid)))
    for i, eps_i in enumerate(eps_imag_grid):
        for j, eps_r in enumerate(eps_real_grid):
            res = _residual_vector(
                [eps_r, eps_i], False, channel_depth_m,
                theta_deg_obs, sigma0_vv_db_obs, sigma0_hh_db_obs, phi_hhvv_rad_obs,
                cfg, phase_weight_deg_equiv,
            )
            rmse_grid[i, j] = np.sqrt(np.mean(res ** 2))

    return {
        "eps_real_grid": eps_real_grid,
        "eps_imag_grid": eps_imag_grid,
        "rmse_grid": rmse_grid,
    }

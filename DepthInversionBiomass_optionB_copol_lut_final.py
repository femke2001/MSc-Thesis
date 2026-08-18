from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator
except Exception as exc:  # pragma: no cover
    RegularGridInterpolator = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None



# =============================================================================
# Physical constants and adopted calibration coefficients
# =============================================================================
C0 = 299_792_458.0
BIOMASS_FREQ_GHZ = 0.435
BIOMASS_FREQ_HZ = BIOMASS_FREQ_GHZ * 1e9
BIOMASS_WAVELENGTH_M = C0 / BIOMASS_FREQ_HZ
EPS0 = 8.854187817e-12


_MAX_OSV_TIME_GAP_S = 60.0


ADOPTED_BETA = 2.72
ADOPTED_A = 1.0


# WGS84 ellipsoid constants (perpendicular-baseline geometry)
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)



def _to_numpy(arr):
    if arr is None:
        return None
    if hasattr(arr, "values"):
        return np.asarray(arr.values)
    return np.asarray(arr)

def extract_radiometry(product):
    sigma0 = product.radiometry_sigmaNought
    gamma0 = product.radiometry_gammaNought
    return sigma0, gamma0

def extract_geometry(product):
    lat = product.geometry_latitude
    lon = product.geometry_longitude
    inc = product.geometry_incidenceAngle
    return lat, lon, inc

def _to_linear_sigma(sigma):
    """Convert sigma0 to linear power if array looks like dB."""
    x = np.asarray(sigma, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return x
    if np.nanmedian(x[finite]) < -0.5 or np.nanmin(x[finite]) < -1.0:
        return 10.0 ** (x / 10.0)
    return x



def _to_float_array(x):
    return np.asarray(x, dtype=float)

def _resample_to_shape(arr, target_shape):
    from scipy.ndimage import zoom

    a = _to_float_array(arr)
    if a.shape == tuple(target_shape):
        return a
    if a.ndim == 0:
        return np.full(target_shape, float(a), dtype=float)
    if a.ndim == 1:
        if a.shape[0] == target_shape[1]:
            return np.tile(a[None, :], (target_shape[0], 1))
        if a.shape[0] == target_shape[0]:
            return np.tile(a[:, None], (1, target_shape[1]))
        x_old = np.linspace(0.0, 1.0, a.shape[0])
        x_new = np.linspace(0.0, 1.0, target_shape[1])
        a_r = np.interp(x_new, x_old, a)
        return np.tile(a_r[None, :], (target_shape[0], 1))
    if a.ndim != 2:
        a = a.reshape(a.shape[0], a.shape[1])
    zf = (target_shape[0] / a.shape[0], target_shape[1] / a.shape[1])
    return zoom(a, zf, order=1)

def to_linear_sigma(sigma):
    x = np.asarray(sigma, dtype=float)
    finite = np.isfinite(x)
    if not np.any(finite):
        return x
    if np.nanmedian(x[finite]) < -0.5 or np.nanmin(x[finite]) < -1.0:
        return 10.0 ** (x / 10.0)
    return x

def lee_filter(img, size=5):
    from scipy.ndimage import uniform_filter

    img = np.asarray(img, dtype=float)
    mean = uniform_filter(img, size=size, mode="nearest")
    mean_sq = uniform_filter(img * img, size=size, mode="nearest")
    var = np.maximum(mean_sq - mean * mean, 0.0)
    noise_var = np.nanmedian(var)
    w = var / (var + noise_var + 1e-12)
    return mean + w * (img - mean)

def topp_dielectric(mv):
    mv = np.asarray(mv, dtype=float)
    eps = 3.03 + 9.3 * mv + 146.0 * (mv ** 2) - 76.7 * (mv ** 3)
    return np.clip(eps, 1.05, 80.0)

def debye_water_permittivity(freq_ghz=BIOMASS_FREQ_GHZ, temp_c=20.0, salinity_ppt=0.0):
    freq_hz = float(freq_ghz) * 1e9
    eps_static = 80.1 - 0.37 * (temp_c - 20.0) - 2.6 * salinity_ppt
    eps_inf = 4.9
    tau = 1.1e-11 * np.exp(0.005346 * (25.0 - temp_c))
    sigma_s_per_m = max(0.0, 0.001 * (1.0 + 0.2 * salinity_ppt))
    wt = 2.0 * np.pi * freq_hz * tau
    real_part = eps_inf + (eps_static - eps_inf) / (1.0 + wt ** 2)
    imag_part = ((eps_static - eps_inf) * wt / (1.0 + wt ** 2)) + sigma_s_per_m / (2.0 * np.pi * freq_hz * EPS0)
    return real_part + 1j * imag_part

def dobson_dielectric_wrapper(mv, sand_pct=85.0, clay_pct=5.0, pb=1.3, freq_ghz=BIOMASS_FREQ_GHZ, temp_c=20.0):
    from dobson_dielectric import dobson_moisture_to_dielectric

    eps_complex = dobson_moisture_to_dielectric(
        mv=np.asarray(mv, dtype=float),
        sand_pct=float(sand_pct),
        clay_pct=float(clay_pct),
        pb=float(pb),
        ps=2.65,
        freq_ghz=float(freq_ghz),
        temp_c=float(temp_c),
    )
    return np.real(eps_complex)

def crim_dielectric(mv, porosity=None, pb=1.3, ps=2.65, eps_solid=3.5, freq_ghz=BIOMASS_FREQ_GHZ, temp_c=20.0, salinity_ppt=0.0):
    mv = np.asarray(mv, dtype=float)
    mv = np.clip(mv, 0.0, 0.6)
    if porosity is None:
        porosity = 1.0 - float(pb) / float(ps)
    phi = float(np.clip(porosity, 0.2, 0.7))
    mv_eff = np.clip(mv, 0.0, phi)
    eps_air = 1.0
    eps_w = debye_water_permittivity(freq_ghz=freq_ghz, temp_c=temp_c, salinity_ppt=salinity_ppt)
    n_eff = ((1.0 - phi) * np.sqrt(eps_solid) + (phi - mv_eff) * np.sqrt(eps_air) + mv_eff * np.sqrt(eps_w))
    return np.real(n_eff ** 2)

def moisture_to_dielectric(mv, model="dobson", **kwargs):
    model = str(model).lower()
    if model == "dobson":
        return dobson_dielectric_wrapper(mv, **kwargs)
    if model == "crim":
        return crim_dielectric(mv, **kwargs)
    if model == "topp":
        return topp_dielectric(mv)
    raise ValueError(f"Unsupported dielectric model: {model}")

def fresnel_coefficients(eps_r, theta_deg):
    eps_r = np.asarray(eps_r, dtype=float)
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    sin_t = np.sin(theta)
    cos_t = np.maximum(np.cos(theta), 1e-8)
    root = np.sqrt(np.maximum(eps_r - sin_t ** 2, 1e-8))
    r_h = (cos_t - root) / np.maximum(cos_t + root, 1e-8)
    r_v = (eps_r * cos_t - root) / np.maximum(eps_r * cos_t + root, 1e-8)
    return r_h, r_v

def roughness_gain_from_ks(ks, beta=ADOPTED_BETA, exponent=ADOPTED_A):
    """
    Monotonic roughness-activation term:

        G_r(ks) = 1 - exp(-beta * ks^exponent)

    Defaults to the calibrated (beta, a) pair fitted against the normalized
    I2EM roughness response (Appendix: roughness_gain_calibration), not the
    original ungrounded placeholder (beta=1.3, a=1.0). This is still an
    empirical activation term, not a full rough-surface scattering solution
    -- see that appendix for the fit's actual shape mismatch at low/high ks
    and its ~20% sensitivity to the assumed L/s ratio.
    """
    ks = np.maximum(np.asarray(ks, dtype=float), 1e-8)
    return 1.0 - np.exp(-float(beta) * ks ** float(exponent))

def copol_surface_sigma0_proxy(eps_r, theta_deg, ks, pol="VV", gain=1.0, beta=ADOPTED_BETA, exponent=ADOPTED_A, angle_power=1.0):
    """
    Physics-guided co-pol forward proxy:

        sigma0_pp ~ gain * |R_pp(eps, theta)|^2 * G_r(ks) * cos(theta)^angle_power

    where:
      R_pp = Fresnel reflection coefficient (HH or VV)
      G_r  = monotonic roughness-activation term in ks (see roughness_gain_from_ks)

    This is NOT a full rough-surface model like IEM/AIEM/I2EM. It is a
    reduced-parameter LUT forward model with explicit, stated assumptions
    (Model 1 in the thesis). See _v7_i2em.py for the physically-based
    I2EM LUT alternative (Model 2).
    """
    r_h, r_v = fresnel_coefficients(eps_r, theta_deg)
    r = r_v if str(pol).upper() == "VV" else r_h
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    rough = roughness_gain_from_ks(ks, beta=beta, exponent=exponent)
    ang = np.maximum(np.cos(th) ** float(angle_power), 1e-8)
    sigma0 = float(gain) * (np.abs(r) ** 2) * rough * ang
    return np.maximum(sigma0, 1e-12)

def calibrate_copol_gain(
    sigma0_obs_lin,
    theta_deg,
    ks,
    pol="VV",
    eps_reference=3.0,
    beta=ADOPTED_BETA,
    exponent=ADOPTED_A,
    angle_power=1.0,
    mask=None,
):
    sigma0_obs = np.maximum(np.asarray(sigma0_obs_lin, dtype=float), 1e-12)
    shape = sigma0_obs.shape

    theta = _resample_to_shape(theta_deg, shape)
    ks_map = _resample_to_shape(ks, shape)

    pred_ref = copol_surface_sigma0_proxy(
        eps_reference,
        theta,
        ks_map,
        pol=pol,
        gain=1.0,
        beta=beta,
        exponent=exponent,
        angle_power=angle_power,
    )

    valid = (
        np.isfinite(sigma0_obs)
        & np.isfinite(theta)
        & np.isfinite(ks_map)
        & np.isfinite(pred_ref)
        & (pred_ref > 0)
        & (sigma0_obs > 0)
    )

    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    vals = sigma0_obs[valid] / pred_ref[valid]

    if vals.size == 0:
        raise RuntimeError(
            "Could not estimate copol gain: no finite positive calibration samples. "
            "Check sigma0, theta, ks, and mask."
        )

    return float(np.nanmedian(vals))

def calibrate_dual_pol_gain(
    sigma0_hh_lin,
    sigma0_vv_lin,
    theta_deg,
    eps_reference=3.0,
    ks_reference=0.5,
    beta=ADOPTED_BETA,
    exponent=ADOPTED_A,
    angle_power=1.0,
    mask=None,
):
    """
    Calibrate gain_hh and gain_vv independently against a fixed reference
    (eps_reference, ks_reference), the same median-ratio approach used for
    a single channel.

    Note: unlike the single-channel calibration above, there is no
    per-pixel ks map available yet at this stage (ks is one of the two
    unknowns about to be solved for jointly), so calibration necessarily
    uses a single scalar ks_reference rather than a measured ks map. If you
    have an independent prior estimate of typical scene ks (e.g. the
    percentile-roughness proxy, kept only as a QA reference -- NOT fed into
    the inversion itself by default), pass its scene median as
    ks_reference.
    """
    gain_hh = calibrate_copol_gain(
        sigma0_hh_lin, theta_deg, ks_reference, pol="HH",
        eps_reference=eps_reference, beta=beta, exponent=exponent,
        angle_power=angle_power, mask=mask,
    )
    gain_vv = calibrate_copol_gain(
        sigma0_vv_lin, theta_deg, ks_reference, pol="VV",
        eps_reference=eps_reference, beta=beta, exponent=exponent,
        angle_power=angle_power, mask=mask,
    )
    return float(gain_hh), float(gain_vv)

def mv_grid_from_dielectric_model(model="dobson", mv_min=0.0, mv_max=0.35, n_mv=351, **kwargs):
    mv_grid = np.linspace(float(mv_min), float(mv_max), int(n_mv), dtype=float)
    eps_grid = moisture_to_dielectric(mv_grid, model=model, **kwargs)
    return mv_grid, np.asarray(eps_grid, dtype=float)

def invert_moisture_from_copol_lut(
    sigma0_obs_lin,
    theta_deg,
    ks,
    pol="VV",
    gain=1.0,
    dielectric_model="dobson",
    mv_min=0.0,
    mv_max=0.35,
    n_mv=351,
    beta=ADOPTED_BETA,
    exponent=ADOPTED_A,
    angle_power=1.0,
    **dielectric_kwargs,
):
    """
    Direct mv LUT inversion via:
        mv -> epsilon'(mv) -> sigma0_pp(epsilon', theta, ks)

    Single co-pol channel, dielectric model explicit, roughness and
    calibration explicit. ks is assumed known (see the joint HH+VV
    inversion below for the two-unknown alternative).
    """
    sigma0 = np.maximum(np.asarray(sigma0_obs_lin, dtype=float), 1e-12)
    shape = sigma0.shape
    theta = _resample_to_shape(theta_deg, shape)
    ks_map = _resample_to_shape(ks, shape)

    mv_grid, eps_grid = mv_grid_from_dielectric_model(
        model=dielectric_model, mv_min=mv_min, mv_max=mv_max, n_mv=n_mv, **dielectric_kwargs,
    )

    pred_stack = np.stack([
        copol_surface_sigma0_proxy(
            eps_g, theta, ks_map, pol=pol, gain=gain,
            beta=beta, exponent=exponent, angle_power=angle_power,
        )
        for eps_g in eps_grid
    ], axis=0)

    obs_log = np.log10(sigma0)[None, :, :]
    pred_log = np.log10(np.maximum(pred_stack, 1e-12))
    idx = np.argmin(np.abs(pred_log - obs_log), axis=0)
    mv_est = mv_grid[idx]
    eps_est = eps_grid[idx]
    return mv_est, eps_est

def invert_eps_ks_from_hh_vv_lut(
    sigma0_hh_lin,
    sigma0_vv_lin,
    theta_deg,
    gain_hh=1.0,
    gain_vv=1.0,
    eps_min=3.0,     # NOTE: does not match thesis Table nominal range (2.5-8.0) -- confirm before citing
    eps_max=25.0,
    n_eps=60,
    ks_min=0.05,     # NOTE: does not match thesis Table nominal range (0.046-0.456) -- confirm before citing
    ks_max=3.0,
    n_ks=60,
    beta=ADOPTED_BETA,
    exponent=ADOPTED_A,
    angle_power=1.0,
    weight_hh=1.0,
    weight_vv=1.0,
):
    """
    Joint double-LUT inversion of (epsilon', ks) from co-registered
    sigma0_HH and sigma0_VV, using the same forward-model proxy as above
    (copol_surface_sigma0_proxy) evaluated for both polarizations.

    Two channels, two unknowns is only well-posed where HH and VV respond
    to (epsilon', ks) in sufficiently different directions -- see the
    companion conditioning study for how to check this BEFORE trusting
    pixel-level retrievals. This function returns a per-pixel conditioning
    diagnostic (cost_gap_map) so poorly-determined pixels can be flagged or
    masked without re-running that separate study.


    Returns
    -------
    eps_est, ks_est, cost_best, cost_gap_map
    """
    sigma0_hh = np.maximum(np.asarray(sigma0_hh_lin, dtype=float), 1e-12)
    sigma0_vv = np.maximum(np.asarray(sigma0_vv_lin, dtype=float), 1e-12)
    shape = sigma0_hh.shape

    if sigma0_vv.shape != shape:
        raise ValueError(
            f"sigma0_hh shape {shape} and sigma0_vv shape {sigma0_vv.shape} must match; "
            "resample/crop both to a common grid before calling this function."
        )

    theta = _resample_to_shape(theta_deg, shape)

    obs_hh_log = np.log10(sigma0_hh)
    obs_vv_log = np.log10(sigma0_vv)

    eps_grid = np.linspace(float(eps_min), float(eps_max), int(n_eps), dtype=float)
    ks_grid = np.linspace(float(ks_min), float(ks_max), int(n_ks), dtype=float)

    best_cost = np.full(shape, np.inf, dtype=float)
    second_best_cost = np.full(shape, np.inf, dtype=float)
    best_eps = np.full(shape, np.nan, dtype=float)
    best_ks = np.full(shape, np.nan, dtype=float)

    for eps_g in eps_grid:
        for ks_g in ks_grid:
            pred_hh = copol_surface_sigma0_proxy(
                eps_g, theta, ks_g, pol="HH", gain=gain_hh,
                beta=beta, exponent=exponent, angle_power=angle_power,
            )
            pred_vv = copol_surface_sigma0_proxy(
                eps_g, theta, ks_g, pol="VV", gain=gain_vv,
                beta=beta, exponent=exponent, angle_power=angle_power,
            )
            pred_hh_log = np.log10(np.maximum(pred_hh, 1e-12))
            pred_vv_log = np.log10(np.maximum(pred_vv, 1e-12))

            cost = (
                float(weight_hh) * (pred_hh_log - obs_hh_log) ** 2
                + float(weight_vv) * (pred_vv_log - obs_vv_log) ** 2
            )

            improved = cost < best_cost
            displaced_to_second = improved & (best_cost < second_best_cost)
            second_best_cost[displaced_to_second] = best_cost[displaced_to_second]

            not_improved_but_better_than_second = (~improved) & (cost < second_best_cost)
            second_best_cost[not_improved_but_better_than_second] = cost[
                not_improved_but_better_than_second
            ]

            best_eps[improved] = eps_g
            best_ks[improved] = ks_g
            best_cost[improved] = cost[improved]

    cost_gap_map = (second_best_cost - best_cost) / np.maximum(best_cost, 1e-8)

    return best_eps, best_ks, best_cost, cost_gap_map

def mv_from_eps_lut(eps_est, dielectric_model="dobson", mv_min=0.0, mv_max=0.35, n_mv=351, **dielectric_kwargs):
    """
    Given a retrieved epsilon' map (from the joint HH+VV inversion, which
    is dielectric-model-agnostic), map it back to volumetric moisture mv
    under a chosen dielectric model, via nearest-match on a 1D mv->eps'(mv)
    grid. Decouples "which epsilon' did the radar see" (answered once,
    jointly, from HH+VV) from "what moisture would that imply under
    dielectric model X" (answered per model, cheaply).
    """
    eps_est = np.asarray(eps_est, dtype=float)
    mv_grid, eps_grid = mv_grid_from_dielectric_model(
        model=dielectric_model, mv_min=mv_min, mv_max=mv_max, n_mv=n_mv, **dielectric_kwargs
    )
    idx = np.argmin(np.abs(eps_grid[:, None, None] - eps_est[None, :, :]), axis=0)
    return mv_grid[idx]

def vertical_wavenumber_surface(theta_deg, wavelength_m=BIOMASS_WAVELENGTH_M, b_perp_m=100.0, r_slant_m=800000.0):
    theta = np.deg2rad(np.asarray(theta_deg, dtype=float))
    return (4.0 * np.pi / float(wavelength_m)) * (
        float(b_perp_m) / (float(r_slant_m) * np.maximum(np.sin(theta), 1e-8))
    )

def vertical_wavenumber_volume(eps_r, theta_deg, wavelength_m=BIOMASS_WAVELENGTH_M, b_perp_m=100.0, r_slant_m=800000.0):
    """
    Volume-corrected vertical wavenumber inside the medium:

        kz_vol = kz_surface * [ eps_r cos(theta) / sqrt(eps_r - sin^2(theta)) ]
    """
    eps_arr = np.asarray(eps_r, dtype=float)
    theta_grid = _resample_to_shape(np.asarray(theta_deg, dtype=float), eps_arr.shape)
    theta = np.deg2rad(theta_grid)
    kz_surface = vertical_wavenumber_surface(
        theta_grid, wavelength_m=wavelength_m, b_perp_m=b_perp_m, r_slant_m=r_slant_m,
    )
    denom = np.sqrt(np.maximum(eps_arr - np.sin(theta) ** 2, 1e-8))
    numer = eps_arr * np.cos(theta)
    return kz_surface * (numer / denom)

def effective_depth_from_gamma(
    gamma_mag,
    eps_r,
    theta_deg,
    wavelength_m=BIOMASS_WAVELENGTH_M,
    b_perp_m=100.0,
    r_slant_m=800000.0,
    use_volume_kz=True,
    model="exponential",   # "exponential" or "gaussian"
    gamma_min=0.4,
    max_depth_m=None,
):
    """
    Convert coherence magnitude to an effective depth scale.

    model="exponential" -> d_eff = -ln|gamma| / |kz_vol|   (thesis Eq. depth_exp)
    model="gaussian"     -> d_eff = sqrt(|gamma|^-2 - 1) / |kz_vol|  (reference case)

    gamma_min masks out low-coherence pixels before depth is computed.
    """
    gamma = np.asarray(gamma_mag, dtype=float)
    gamma = np.clip(gamma, 1e-6, 0.999999)

    kz_surface = vertical_wavenumber_surface(
        theta_deg, wavelength_m=wavelength_m, b_perp_m=b_perp_m, r_slant_m=r_slant_m,
    )
    kz_vol = vertical_wavenumber_volume(
        eps_r, theta_deg, wavelength_m=wavelength_m, b_perp_m=b_perp_m, r_slant_m=r_slant_m,
    )
    kz_used = kz_vol if use_volume_kz else kz_surface

    depth = np.full(gamma.shape, np.nan, dtype=float)
    valid = gamma >= float(gamma_min)

    if model == "gaussian":
        depth[valid] = (
            np.sqrt(np.maximum(gamma[valid] ** (-2.0) - 1.0, 0.0))
            / np.maximum(np.abs(kz_used[valid]), 1e-8)
        )
    elif model == "exponential":
        depth[valid] = -np.log(gamma[valid]) / np.maximum(np.abs(kz_used[valid]), 1e-8)
    else:
        raise ValueError("model must be 'gaussian' or 'exponential'")

    if max_depth_m is not None:
        depth = np.clip(depth, 0.0, float(max_depth_m))

    return depth, kz_used, kz_surface, kz_vol

def retrieve_mv_and_depth_from_copol(
    sigma0_copol,
    gamma_mag,
    theta_deg,
    ks,
    pol="VV",
    dielectric_model="dobson",
    gain=None,
    gain_eps_reference=3.0,
    gain_mask=None,
    wavelength_m=BIOMASS_WAVELENGTH_M,
    b_perp_m=100.0,
    r_slant_m=800000.0,
    mv_min=0.0,
    mv_max=0.35,
    n_mv=351,
    beta=ADOPTED_BETA,
    exponent=ADOPTED_A,
    angle_power=1.0,
    speckle_filter_size=None,
    use_volume_kz=True,
    depth_model="exponential",
    gamma_min=0.4,
    max_depth_m=None,
    **dielectric_kwargs,
):
    """
    Single-channel (ks assumed known) end-to-end retrieval: sigma0 -> mv,
    eps' -> depth. Kept for comparison against the joint HH+VV path below;
    NOT the primary/nominal retrieval (see thesis Model 1 vs. the
    ks-assumed single-channel predecessor it replaced).
    """
    sigma0_lin = to_linear_sigma(sigma0_copol)
    sigma0_lin = np.maximum(np.asarray(sigma0_lin, dtype=float), 1e-12)

    if speckle_filter_size is not None and int(speckle_filter_size) > 1:
        sigma0_lin = lee_filter(sigma0_lin, size=int(speckle_filter_size))

    if gain is None:
        gain = calibrate_copol_gain(
            sigma0_lin, theta_deg, ks, pol=pol,
            eps_reference=gain_eps_reference, beta=beta, exponent=exponent,
            angle_power=angle_power, mask=gain_mask,
        )

    mv_est, eps_est = invert_moisture_from_copol_lut(
        sigma0_lin, theta_deg, ks, pol=pol, gain=gain,
        dielectric_model=dielectric_model, mv_min=mv_min, mv_max=mv_max, n_mv=n_mv,
        beta=beta, exponent=exponent, angle_power=angle_power, **dielectric_kwargs,
    )

    depth_m, kz_used, kz_surface, kz_vol = effective_depth_from_gamma(
        gamma_mag, eps_est, theta_deg,
        wavelength_m=wavelength_m, b_perp_m=b_perp_m, r_slant_m=r_slant_m,
        use_volume_kz=use_volume_kz, model=depth_model,
        gamma_min=gamma_min, max_depth_m=max_depth_m,
    )

    return {
        "mv_est": mv_est,
        "eps_est": eps_est,
        "depth_m": depth_m,
        "kz_used": kz_used,
        "kz_surface": kz_surface,
        "kz_vol": kz_vol,
        "kz_mode": "volume" if use_volume_kz else "surface",
        "depth_model": depth_model,
        "gamma_min": float(gamma_min),
        "max_depth_m": None if max_depth_m is None else float(max_depth_m),
        "gain_used": float(gain),
    }

def retrieve_eps_ks_mv_and_depth_from_hh_vv(
    sigma0_hh,
    sigma0_vv,
    gamma_mag,
    theta_deg,
    dielectric_model="dobson",
    gain_hh=None,
    gain_vv=None,
    gain_eps_reference=3.0,
    gain_ks_reference=0.5,
    gain_mask=None,
    eps_min=2.0,     # NOTE: does not match thesis Table nominal range (2.5-8.0) -- confirm before citing
    eps_max=25.0,
    n_eps=60,
    ks_min=0.05,     # NOTE: does not match thesis Table nominal range (0.046-0.456) -- confirm before citing
    ks_max=0.5,
    n_ks=60,
    beta=ADOPTED_BETA,
    exponent=ADOPTED_A,
    angle_power=1.0,
    weight_hh=1.0,
    weight_vv=1.0,
    wavelength_m=BIOMASS_WAVELENGTH_M,
    b_perp_m=100.0,
    r_slant_m=800000.0,
    mv_min=0.0,
    mv_max=0.35,
    n_mv=351,
    speckle_filter_size=None,
    use_volume_kz=True,
    depth_model="exponential",
    gamma_min=0.4,
    max_depth_m=None,
    **dielectric_kwargs,
):
    """
    Joint HH+VV double-LUT workflow (thesis Model 1): solves for
    (epsilon', ks) jointly from two channels instead of assuming ks and
    inverting epsilon'/mv from one channel.

    Workflow
    --------
    1) convert sigma0_HH, sigma0_VV to linear if needed
    2) optional Lee filter (each channel independently, intensity domain)
    3) calibrate scene-level gain_hh, gain_vv if not provided
    4) jointly invert (epsilon', ks) via 2D LUT search over (sigma0_HH, sigma0_VV)
    5) map epsilon' -> mv under the chosen dielectric model (cheap reverse lookup)
    6) compute effective decorrelation depth from coherence using the
       RETRIEVED epsilon' (not an externally assumed one)

    Returns
    -------
    dict with eps_est, ks_est, mv_est, depth_m, cost_best, cost_gap_map,
    kz_used, kz_surface, kz_vol, gain_hh_used, gain_vv_used
    """
    sigma0_hh_lin = np.maximum(np.asarray(to_linear_sigma(sigma0_hh), dtype=float), 1e-12)
    sigma0_vv_lin = np.maximum(np.asarray(to_linear_sigma(sigma0_vv), dtype=float), 1e-12)

    if speckle_filter_size is not None and int(speckle_filter_size) > 1:
        sigma0_hh_lin = lee_filter(sigma0_hh_lin, size=int(speckle_filter_size))
        sigma0_vv_lin = lee_filter(sigma0_vv_lin, size=int(speckle_filter_size))

    if gain_hh is None or gain_vv is None:
        cal_hh, cal_vv = calibrate_dual_pol_gain(
            sigma0_hh_lin, sigma0_vv_lin, theta_deg,
            eps_reference=gain_eps_reference, ks_reference=gain_ks_reference,
            beta=beta, exponent=exponent, angle_power=angle_power, mask=gain_mask,
        )
        gain_hh = cal_hh if gain_hh is None else gain_hh
        gain_vv = cal_vv if gain_vv is None else gain_vv

    eps_est, ks_est, cost_best, cost_gap_map = invert_eps_ks_from_hh_vv_lut(
        sigma0_hh_lin, sigma0_vv_lin, theta_deg,
        gain_hh=gain_hh, gain_vv=gain_vv,
        eps_min=eps_min, eps_max=eps_max, n_eps=n_eps,
        ks_min=ks_min, ks_max=ks_max, n_ks=n_ks,
        beta=beta, exponent=exponent, angle_power=angle_power,
        weight_hh=weight_hh, weight_vv=weight_vv,
    )

    mv_est = mv_from_eps_lut(
        eps_est, dielectric_model=dielectric_model,
        mv_min=mv_min, mv_max=mv_max, n_mv=n_mv, **dielectric_kwargs,
    )

    depth_m, kz_used, kz_surface, kz_vol = effective_depth_from_gamma(
        gamma_mag, eps_est, theta_deg,
        wavelength_m=wavelength_m, b_perp_m=b_perp_m, r_slant_m=r_slant_m,
        use_volume_kz=use_volume_kz, model=depth_model,
        gamma_min=gamma_min, max_depth_m=max_depth_m,
    )

    return {
        "eps_est": eps_est,
        "ks_est": ks_est,
        "mv_est": mv_est,
        "depth_m": depth_m,
        "cost_best": cost_best,
        "cost_gap_map": cost_gap_map,
        "kz_used": kz_used,
        "kz_surface": kz_surface,
        "kz_vol": kz_vol,
        "kz_mode": "volume" if use_volume_kz else "surface",
        "depth_model": depth_model,
        "gamma_min": float(gamma_min),
        "max_depth_m": None if max_depth_m is None else float(max_depth_m),
        "gain_hh_used": float(gain_hh),
        "gain_vv_used": float(gain_vv),
    }


# =============================================================================
# Model 2 (I2EM physical LUT) retrieval
# =============================================================================

def _db(x):
    return 10.0 * np.log10(np.clip(np.asarray(x, dtype=float), 1e-30, None))

def _lin_from_db(x_db):
    return np.power(10.0, np.asarray(x_db, dtype=float) / 10.0)

def _subset_axis(axis, min_val=None, max_val=None):
    axis = np.asarray(axis, dtype=float)
    mask = np.ones(axis.shape, dtype=bool)
    if min_val is not None:
        mask &= axis >= float(min_val)
    if max_val is not None:
        mask &= axis <= float(max_val)
    if not np.any(mask):
        raise ValueError("Selected axis subset is empty. Check min/max settings.")
    return axis[mask]

def _standardise_sigma_array(arr, n_theta, n_ks, n_eps, name):
    arr = np.asarray(arr, dtype=float)
    expected = (n_theta, n_ks, n_eps)
    if arr.shape != expected:
        raise ValueError(
            f"{name} has shape {arr.shape}, expected {expected}. "
            "The required axis order is (theta, ks, eps)."
        )
    return arr

class I2EMLUT:
    """Interpolated I2EM sigma0 lookup table, internally stored in dB."""

    def __init__(self, eps_axis, ks_axis, theta_axis, sigma0_hh, sigma0_vv, sigma0_hv=None, sigma0_units="dB"):
        if RegularGridInterpolator is None:
            raise ImportError(f"scipy is required for interpolation: {_SCIPY_IMPORT_ERROR}")

        self.eps_axis = np.asarray(eps_axis, dtype=float)
        self.ks_axis = np.asarray(ks_axis, dtype=float)
        self.theta_axis = np.asarray(theta_axis, dtype=float)

        if np.any(np.diff(self.eps_axis) <= 0):
            raise ValueError("eps_axis must be strictly increasing.")
        if np.any(np.diff(self.ks_axis) <= 0):
            raise ValueError("ks_axis must be strictly increasing.")
        if np.any(np.diff(self.theta_axis) <= 0):
            raise ValueError("theta_axis must be strictly increasing.")

        n_theta = len(self.theta_axis)
        n_ks = len(self.ks_axis)
        n_eps = len(self.eps_axis)

        sigma0_hh = _standardise_sigma_array(sigma0_hh, n_theta, n_ks, n_eps, "sigma0_hh")
        sigma0_vv = _standardise_sigma_array(sigma0_vv, n_theta, n_ks, n_eps, "sigma0_vv")
        if sigma0_hv is not None:
            sigma0_hv = _standardise_sigma_array(sigma0_hv, n_theta, n_ks, n_eps, "sigma0_hv")

        units = str(sigma0_units).lower()
        if units in ("db", "decibel", "decibels"):
            hh_db = sigma0_hh
            vv_db = sigma0_vv
            hv_db = sigma0_hv
        elif units in ("linear", "lin", "power"):
            hh_db = _db(sigma0_hh)
            vv_db = _db(sigma0_vv)
            hv_db = None if sigma0_hv is None else _db(sigma0_hv)
        else:
            raise ValueError("sigma0_units must be 'dB' or 'linear'.")

        points = (self.theta_axis, self.ks_axis, self.eps_axis)
        self._interp = {
            "hh": RegularGridInterpolator(points, hh_db, bounds_error=False, fill_value=np.nan),
            "vv": RegularGridInterpolator(points, vv_db, bounds_error=False, fill_value=np.nan),
        }
        if hv_db is not None:
            self._interp["hv"] = RegularGridInterpolator(points, hv_db, bounds_error=False, fill_value=np.nan)

    @property
    def has_hv(self):
        return "hv" in self._interp

    def predict_db(self, eps, ks, theta_deg, channel="vv"):
        ch = str(channel).lower()
        if ch not in self._interp:
            raise ValueError(f"Channel {channel!r} is not available in this LUT.")
        eps = np.asarray(eps, dtype=float)
        ks = np.asarray(ks, dtype=float)
        theta = np.asarray(theta_deg, dtype=float)
        eps_b, ks_b, th_b = np.broadcast_arrays(eps, ks, theta)
        pts = np.column_stack([th_b.ravel(), ks_b.ravel(), eps_b.ravel()])
        return self._interp[ch](pts).reshape(eps_b.shape)

    def predict_linear(self, eps, ks, theta_deg, channel="vv"):
        return _lin_from_db(self.predict_db(eps, ks, theta_deg, channel=channel))

def load_i2em_lut_npz(path, sigma0_units="dB"):
    data = np.load(path, allow_pickle=True)
    required = ["eps_axis", "ks_axis", "theta_axis", "sigma0_hh", "sigma0_vv"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"I2EM LUT is missing required keys: {missing}")

    # If the file stores a scalar sigma0_units string and the user did not
    # override it meaningfully, use the stored value.
    if "sigma0_units" in data and sigma0_units is None:
        sigma0_units = str(data["sigma0_units"].item())

    return I2EMLUT(
        eps_axis=data["eps_axis"],
        ks_axis=data["ks_axis"],
        theta_axis=data["theta_axis"],
        sigma0_hh=data["sigma0_hh"],
        sigma0_vv=data["sigma0_vv"],
        sigma0_hv=data["sigma0_hv"] if "sigma0_hv" in data else None,
        sigma0_units=sigma0_units or "dB",
    )

def invert_eps_ks_from_i2em_lut(
    sigma0_hh,
    sigma0_vv,
    theta_deg,
    i2em_lut,
    *,
    sigma0_hv=None,
    channels=("hh", "vv"),
    sigma0_input_units="linear",
    eps_min=2.5,
    eps_max=8.0,
    ks_min=0.046,
    ks_max=0.456,
    n_eps=None,
    n_ks=None,
    weights=None,
    ks_prior=None,
    sigma_ks_prior=None,
    eps_prior=None,
    sigma_eps_prior=None,
    theta_bin_step=0.25,
    block_size=200000,
):
    """Joint LUT inversion of epsilon' and ks from HH/VV/(HV) backscatter."""
    channels = tuple(str(c).lower() for c in channels)
    for c in channels:
        if c not in ("hh", "vv", "hv"):
            raise ValueError("channels must contain only 'hh', 'vv' and optionally 'hv'.")
        if c == "hv" and (sigma0_hv is None or not i2em_lut.has_hv):
            raise ValueError("HV requested, but sigma0_hv input or sigma0_hv LUT is missing.")

    units = str(sigma0_input_units).lower()
    if units in ("db", "decibel", "decibels"):
        hh_db = np.asarray(sigma0_hh, dtype=float)
        vv_db = np.asarray(sigma0_vv, dtype=float)
        hv_db = None if sigma0_hv is None else np.asarray(sigma0_hv, dtype=float)
    elif units in ("linear", "lin", "power"):
        hh_db = _db(sigma0_hh)
        vv_db = _db(sigma0_vv)
        hv_db = None if sigma0_hv is None else _db(sigma0_hv)
    else:
        raise ValueError("sigma0_input_units must be 'linear' or 'dB'.")

    shape = hh_db.shape
    vv_db = np.broadcast_to(vv_db, shape)
    theta_arr = np.broadcast_to(np.asarray(theta_deg, dtype=float), shape)

    obs = {"hh": hh_db, "vv": vv_db}
    if hv_db is not None:
        obs["hv"] = np.broadcast_to(hv_db, shape)

    eps_axis = _subset_axis(i2em_lut.eps_axis, eps_min, eps_max)
    ks_axis = _subset_axis(i2em_lut.ks_axis, ks_min, ks_max)
    if n_eps is not None:
        eps_axis = np.linspace(eps_axis.min(), eps_axis.max(), int(n_eps))
    if n_ks is not None:
        ks_axis = np.linspace(ks_axis.min(), ks_axis.max(), int(n_ks))

    EPS, KS = np.meshgrid(eps_axis, ks_axis, indexing="xy")
    eps_flat = EPS.ravel()
    ks_flat = KS.ravel()
    n_cand = eps_flat.size


    _BLOCK_MEM_BUDGET_BYTES = 256 * 1024**2            # ~256 MiB transient/block
    _bytes_per_pixel_row = 16 * max(int(n_cand), 1)    # ~16 B/elem across temps
    block_size = max(1, min(int(block_size),
                            _BLOCK_MEM_BUDGET_BYTES // _bytes_per_pixel_row))

    if weights is None:
        weights = {c: 1.0 for c in channels}
    else:
        weights = {c: float(weights.get(c, 1.0)) for c in channels}

    valid = np.isfinite(theta_arr)
    for c in channels:
        valid &= np.isfinite(obs[c])

    eps_est = np.full(shape, np.nan, dtype=float)
    ks_est = np.full(shape, np.nan, dtype=float)
    best_cost = np.full(shape, np.nan, dtype=float)
    cost_gap = np.full(shape, np.nan, dtype=float)
    eps_second = np.full(shape, np.nan, dtype=float)
    ks_second = np.full(shape, np.nan, dtype=float)

    valid_idx = np.where(valid.ravel())[0]
    if valid_idx.size == 0:
        return {
            "eps_est": eps_est,
            "ks_est": ks_est,
            "best_cost": best_cost,
            "cost_gap_map": cost_gap,
            "eps_second": eps_second,
            "ks_second": ks_second,
            "eps_axis_used": eps_axis,
            "ks_axis_used": ks_axis,
        }

    theta_flat = theta_arr.ravel()
    if theta_bin_step is None or theta_bin_step <= 0:
        theta_bins = theta_flat[valid_idx]
    else:
        theta_bins = np.round(theta_flat[valid_idx] / float(theta_bin_step)) * float(theta_bin_step)

    unique_bins = np.unique(theta_bins[np.isfinite(theta_bins)])

    eps_f = eps_est.ravel()
    ks_f = ks_est.ravel()
    best_f = best_cost.ravel()
    gap_f = cost_gap.ravel()
    eps2_f = eps_second.ravel()
    ks2_f = ks_second.ravel()
    obs_f = {c: obs[c].ravel() for c in channels}

    ks_prior_f = None if ks_prior is None else np.broadcast_to(np.asarray(ks_prior, dtype=float), shape).ravel()
    eps_prior_f = None if eps_prior is None else np.broadcast_to(np.asarray(eps_prior, dtype=float), shape).ravel()

    for th in unique_bins:
        pix_all = valid_idx[theta_bins == th]
        pred = {c: i2em_lut.predict_db(eps_flat, ks_flat, th, channel=c) for c in channels}
        for c in channels:
            if not np.any(np.isfinite(pred[c])):
                raise RuntimeError(
                    f"LUT interpolation returned all NaN for theta={th}. "
                    "Check that theta/eps/ks search ranges are inside the LUT axes."
                )

        for start in range(0, pix_all.size, int(block_size)):
            pix = pix_all[start:start + int(block_size)]
            cost = np.zeros((pix.size, n_cand), dtype=np.float32)
            for c in channels:
                # Same arithmetic as before (float64 subtraction -> float32
                # round -> float32 square -> float32 weight -> float32 sum), but
                # done in place so only one full-size float32 temporary is live
                # alongside cost. The float64 residual is released before the
                # square, roughly halving the transient peak. Bit-identical to
                # `cost += np.float32(weights[c]) * (resid.astype(np.float32) ** 2)`.
                resid = obs_f[c][pix, None] - pred[c][None, :]
                r32 = resid.astype(np.float32)
                del resid
                r32 *= r32
                if weights[c] != 1.0:
                    r32 *= np.float32(weights[c])
                cost += r32
                del r32

            if ks_prior_f is not None and sigma_ks_prior is not None and float(sigma_ks_prior) > 0:
                cost += (((ks_flat[None, :] - ks_prior_f[pix, None]) / float(sigma_ks_prior)) ** 2).astype(np.float32)

            if eps_prior_f is not None and sigma_eps_prior is not None and float(sigma_eps_prior) > 0:
                cost += (((eps_flat[None, :] - eps_prior_f[pix, None]) / float(sigma_eps_prior)) ** 2).astype(np.float32)

            cost[~np.isfinite(cost)] = np.inf
            idx_best = np.argmin(cost, axis=1)
            val_best = cost[np.arange(pix.size), idx_best]

            eps_f[pix] = eps_flat[idx_best]
            ks_f[pix] = ks_flat[idx_best]
            best_f[pix] = val_best

            if n_cand > 1:
                cost[np.arange(pix.size), idx_best] = np.inf
                idx_second = np.argmin(cost, axis=1)
                val_second = cost[np.arange(pix.size), idx_second]
                eps2_f[pix] = eps_flat[idx_second]
                ks2_f[pix] = ks_flat[idx_second]
                gap_f[pix] = val_second - val_best

    return {
        "eps_est": eps_est,
        "ks_est": ks_est,
        "best_cost": best_cost,
        "cost_gap_map": cost_gap,
        "eps_second": eps_second,
        "ks_second": ks_second,
        "eps_axis_used": eps_axis,
        "ks_axis_used": ks_axis,
        "channels": channels,
    }

def retrieve_eps_ks_mv_and_depth_from_i2em_lut(
    sigma0_hh,
    sigma0_vv,
    gamma_mag,
    theta_deg,
    i2em_lut,
    *,
    sigma0_hv=None,
    channels=("hh", "vv"),
    sigma0_input_units="linear",
    eps_min=2.5,
    eps_max=8.0,
    ks_min=0.046,
    ks_max=0.456,
    n_eps=None,
    n_ks=None,
    weights=None,
    ks_prior=None,
    sigma_ks_prior=None,
    eps_prior=None,
    sigma_eps_prior=None,
    theta_bin_step=0.25,
    dielectric_model="dobson",
    mv_min=0.0,
    mv_max=0.35,
    n_mv=351,
    wavelength_m=BIOMASS_WAVELENGTH_M,
    b_perp_m=100.0,
    r_slant_m=800000.0,
    use_volume_kz=True,
    depth_model="exponential",
    gamma_min=0.1,
    max_depth_m=None,
    **dielectric_kwargs,
):
    """End-to-end I2EM-LUT retrieval: epsilon', ks, mv and depth."""
    inv = invert_eps_ks_from_i2em_lut(
        sigma0_hh=sigma0_hh,
        sigma0_vv=sigma0_vv,
        sigma0_hv=sigma0_hv,
        theta_deg=theta_deg,
        i2em_lut=i2em_lut,
        channels=channels,
        sigma0_input_units=sigma0_input_units,
        eps_min=eps_min,
        eps_max=eps_max,
        ks_min=ks_min,
        ks_max=ks_max,
        n_eps=n_eps,
        n_ks=n_ks,
        weights=weights,
        ks_prior=ks_prior,
        sigma_ks_prior=sigma_ks_prior,
        eps_prior=eps_prior,
        sigma_eps_prior=sigma_eps_prior,
        theta_bin_step=theta_bin_step,
    )

    eps_est = inv["eps_est"]
    mv_est = mv_from_eps_lut(
        eps_est,
        dielectric_model=dielectric_model,
        mv_min=mv_min,
        mv_max=mv_max,
        n_mv=n_mv,
        **dielectric_kwargs,
    )

    depth_m, kz_used, kz_surface, kz_vol = effective_depth_from_gamma(
        gamma_mag=gamma_mag,
        eps_r=eps_est,
        theta_deg=theta_deg,
        wavelength_m=wavelength_m,
        b_perp_m=b_perp_m,
        r_slant_m=r_slant_m,
        use_volume_kz=use_volume_kz,
        model=depth_model,
        gamma_min=gamma_min,
        max_depth_m=max_depth_m,
    )

    inv.update(
        {
            "mv_est": mv_est,
            "depth_m": depth_m,
            "kz_used": kz_used,
            "kz_surface": kz_surface,
            "kz_vol": kz_vol,
        }
    )
    return inv

def retrieval_quality_mask(
    result,
    *,
    eps_min=None,
    eps_max=None,
    ks_min=None,
    ks_max=None,
    max_cost=None,
    min_cost_gap=None,
    edge_margin=0,
):
    """Create a boolean mask for reliable retrieval pixels."""
    eps = np.asarray(result["eps_est"], dtype=float)
    ks = np.asarray(result["ks_est"], dtype=float)
    mask = np.isfinite(eps) & np.isfinite(ks)

    if eps_min is not None:
        mask &= eps > float(eps_min)
    if eps_max is not None:
        mask &= eps < float(eps_max)
    if ks_min is not None:
        mask &= ks > float(ks_min)
    if ks_max is not None:
        mask &= ks < float(ks_max)
    if max_cost is not None:
        mask &= np.asarray(result["best_cost"], dtype=float) <= float(max_cost)
    if min_cost_gap is not None:
        mask &= np.asarray(result["cost_gap_map"], dtype=float) >= float(min_cost_gap)

    if edge_margin and int(edge_margin) > 0:
        m = int(edge_margin)
        mask[:m, :] = False
        mask[-m:, :] = False
        mask[:, :m] = False
        mask[:, -m:] = False

    return mask

def bound_hit_summary(result, eps_min, eps_max, ks_min, ks_max, tol=1e-9):
    """Return simple percentages of pixels stuck on LUT bounds."""
    eps = np.asarray(result["eps_est"], dtype=float)
    ks = np.asarray(result["ks_est"], dtype=float)
    valid = np.isfinite(eps) & np.isfinite(ks)
    if not np.any(valid):
        return {"n_valid": 0}
    return {
        "n_valid": int(np.sum(valid)),
        "eps_min_pct": 100.0 * float(np.mean(eps[valid] <= eps_min + tol)),
        "eps_max_pct": 100.0 * float(np.mean(eps[valid] >= eps_max - tol)),
        "ks_min_pct": 100.0 * float(np.mean(ks[valid] <= ks_min + tol)),
        "ks_max_pct": 100.0 * float(np.mean(ks[valid] >= ks_max - tol)),
    }




def _strip_ns(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag

def _parse_time_value(value):
    if value is None:
        return None
    txt = str(value).strip()
    if not txt:
        return None
    if "=" in txt:
        txt = txt.split("=", 1)[1].strip()
    txt = txt.replace("Z", "")
    try:
        return datetime.fromisoformat(txt)
    except Exception:
        return None

def _product_reference_time(product):
    candidates = [
        "global_referenceAzimuthTime",
        "global_startTime",
        "global_stopTime",
        "ascendingNodeDate",
        "startTimeFromAscendingNode",
        "completionTimeFromAscendingNode",
    ]
    for name in candidates:
        if hasattr(product, name):
            dt = _parse_time_value(getattr(product, name))
            if dt is not None:
                return dt, name
    return None, None

def _first_scalar_from_names(product, names):
    for name in names:
        if hasattr(product, name):
            try:
                value = _to_numpy(getattr(product, name))
                if value is None:
                    continue
                array = np.asarray(value, dtype=float)
                if array.size == 0:
                    continue
                scalar = float(np.nanmedian(array))
                if np.isfinite(scalar):
                    return scalar, name
            except Exception:
                pass
    return None, None

def _vector_from_component_names(product, component_sets):
    for x_name, y_name, z_name in component_sets:
        x_val, x_src = _first_scalar_from_names(product, [x_name])
        y_val, y_src = _first_scalar_from_names(product, [y_name])
        z_val, z_src = _first_scalar_from_names(product, [z_name])
        if all(value is not None and np.isfinite(value) for value in (x_val, y_val, z_val)):
            return np.array([x_val, y_val, z_val], dtype=float), f"{x_src}/{y_src}/{z_src}"
    return None, None

def _vector_from_vector_names(product, names):
    for name in names:
        if not hasattr(product, name):
            continue
        try:
            value = _to_numpy(getattr(product, name))
            if value is None:
                continue
            array = np.asarray(value, dtype=float)
            if array.size < 3:
                continue
            if array.ndim == 1:
                vector = array[:3]
            elif array.ndim >= 2:
                reshaped = array.reshape(-1, array.shape[-1]) if array.shape[-1] >= 3 else array.reshape(-1, 3)
                if reshaped.shape[-1] < 3:
                    continue
                vector = np.nanmedian(reshaped[:, :3], axis=0)
            else:
                continue
            if np.all(np.isfinite(vector)):
                return np.asarray(vector, dtype=float), name
        except Exception:
            pass
    return None, None

def _orb_xml_path_from_root(root_folder) -> Path:
    """Derive the real orbit-XML path directly from a product's root
    folder, bypassing any reliance on the product object exposing an
    'annotation_orb_file' attribute (which it may not).

    Verified pattern from a real product folder:
        root:  BIO_S1_STA__1S_20260501T164629_20260501T164649_T_G01_M03_C03_T040_F135_02_DS7KUK
        orbit: annotation_coregistered/navigation/
               bio_s1_sta__1s_20260501t164629_20260501t164649_t_g01_m03_c03_t040_f135_orb.xml

    i.e. strip the trailing "_<2-digit-counter>_<dataset-id>" suffix from
    the root folder name, lowercase everything else, append "_orb.xml",
    and look for it under annotation_coregistered/navigation/.

    IMPORTANT -- annotation_coregistered, NOT annotation_primary.
    An STA product folder contains BOTH subdirectories, holding files with
    the SAME name but different contents:

        annotation_coregistered/  -> state vectors for THIS scene
        annotation_primary/       -> state vectors for the STACK REFERENCE
                                     scene (identical in every folder of
                                     the stack)

    Verified on a real C03/C04 stack: the C03 folder's annotation_primary
    file contains 2026-05-04 OSVs (the C04 stack reference), while its
    annotation_coregistered file contains the correct 2026-05-01 OSVs.
    Reading annotation_primary therefore makes every scene in the stack
    return the reference scene's orbit, so B_perp collapses to an
    along-track separation instead of a repeat-pass baseline. This is
    invisible for the reference scene itself, where the two files agree.
    """
    root_folder = Path(root_folder)
    base = re.sub(r"_\d+_[A-Za-z0-9]+$", "", root_folder.name)
    return root_folder / "annotation_coregistered" / "navigation" / f"{base.lower()}_orb.xml"

def _reference_time_from_root_name(root_folder):
    """Extract the scene's actual acquisition midpoint time directly from
    the product folder name (e.g. ..._20260501T164629_20260501T164649_...),
    rather than relying on the product object exposing a start/stop-time
    attribute under one of the guessed candidate names in
    _product_reference_time(). This matters: tested against a real BIOMASS
    orbit file, falling back to the orbit file's own midpoint (when no
    reference time is available at all) picked a state vector ~330 seconds
    / ~2490 km along-track away from the real scene time -- more than
    enough to silently produce a badly wrong baseline with no error or
    warning. The folder name is a much more reliable source than hoping an
    attribute name guess happens to match."""
    if root_folder is None:
        return None
    name = Path(root_folder).name
    m = re.search(r"(\d{8}T\d{6})_(\d{8}T\d{6})", name)
    if not m:
        return None
    try:
        start = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
        stop = datetime.strptime(m.group(2), "%Y%m%dT%H%M%S")
        return start + (stop - start) / 2
    except Exception:
        return None

def _state_vector_from_orbit_xml(product, orb_xml_path=None, ref_time_override=None):
    """orb_xml_path, if given, is used directly (and must exist) --
    this is the preferred path, since it comes from
    _orb_xml_path_from_root() rather than an attribute guess. Falls back
    to the original attribute-based lookup (getattr(product,
    'annotation_orb_file', ...)) only if orb_xml_path isn't provided.

    ref_time_override, if given (e.g. from _reference_time_from_root_name),
    takes priority over _product_reference_time(product)'s attribute-name
    guessing. If NEITHER produces a usable reference time, this prints an
    explicit warning before falling back to the orbit file's own midpoint
    record -- verified against real data to be able to land ~2500 km
    away from the correct scene position, which would otherwise silently
    corrupt the baseline computation with no visible error."""
    if orb_xml_path is not None:
        orb_path = Path(orb_xml_path)
        if not orb_path.exists():
            return None, None, None
    else:
        orb_path = Path(getattr(product, "annotation_orb_file", ""))
        if not orb_path.exists():
            return None, None, None

    try:
        root = ET.parse(str(orb_path)).getroot()
    except Exception:
        return None, None, None

    ref_time, ref_src = (ref_time_override, "root_folder_name") if ref_time_override is not None else _product_reference_time(product)

    records = []
    for elem in root.iter():
        if _strip_ns(elem.tag) != "OSV":
            continue

        values = {}
        for child in list(elem):
            key = _strip_ns(child.tag)
            values[key] = (child.text or "").strip()

        try:
            x = float(values.get("X", "nan"))
            y = float(values.get("Y", "nan"))
            z = float(values.get("Z", "nan"))
            vx = float(values.get("VX", "nan"))
            vy = float(values.get("VY", "nan"))
            vz = float(values.get("VZ", "nan"))
        except Exception:
            continue

        if not all(np.isfinite(v) for v in (x, y, z, vx, vy, vz)):
            continue

        t = _parse_time_value(values.get("UTC") or values.get("TAI") or values.get("UT1"))
        records.append({
            "time": t,
            "pos": np.array([x, y, z], dtype=float),
            "vel": np.array([vx, vy, vz], dtype=float),
        })

    if not records:
        return None, None, None

    idx = len(records) // 2
    if ref_time is not None:
        timed = [r for r in records if r["time"] is not None]
        if timed:
            idx = min(
                range(len(records)),
                key=lambda i: abs((records[i]["time"] - ref_time).total_seconds()) if records[i]["time"] is not None else np.inf,
            )
    else:
        print(
            "WARNING: no reference time available (neither ref_time_override nor a "
            "matching product attribute) -- falling back to the orbit file's own "
            "midpoint record. Verified against real data: this can pick a state "
            "vector several minutes / >1000 km away from the actual scene position, "
            "silently producing a wrong baseline. Pass master_root/secondary_root to "
            "estimate_bperp_from_products() so the reference time can be derived from "
            "the product folder name instead."
        )

    rec = records[idx]

    # --- coverage check --------------------------------------------------
    # min() over |t - ref_time| always returns SOMETHING. If the scene time
    # falls outside the file's coverage the "nearest" record is just the
    # first or last one, which can be days away. Verified on a real stack:
    # reading the wrong orbit file gave a nearest OSV 3 days off, clamped to
    # OSV[0], producing a plausible-looking but 31x-wrong depth map with no
    # error. OSVs here are on a ~1 s grid, so a correct match is sub-second;
    # anything beyond a minute means the wrong orbit file, not a rounding
    # issue. Fail loudly rather than clamp.
    if ref_time is not None and rec["time"] is not None:
        _dt = abs((rec["time"] - ref_time).total_seconds())
        if _dt > _MAX_OSV_TIME_GAP_S:
            _t0 = records[0]["time"]
            _t1 = records[-1]["time"]
            raise RuntimeError(
                f"Nearest OSV in {orb_path.name} is {_dt:.0f} s from the scene "
                f"reference time {ref_time.isoformat()} (record {idx} of "
                f"{len(records)}, at {rec['time'].isoformat()}). File coverage "
                f"is {_t0} -> {_t1}. The scene time is outside this orbit "
                f"file's coverage, i.e. this is the wrong orbit file -- check "
                f"that _orb_xml_path_from_root() points at "
                f"annotation_coregistered/ and not annotation_primary/."
            )

    t_txt = rec["time"].isoformat() if rec["time"] is not None else "unknown_time"
    src = f"{orb_path.parent.parent.name}:OSV[{idx}]@{t_txt}"
    if ref_src is not None:
        src += f" (nearest {ref_src})"
    return rec["pos"], rec["vel"], src

def _scene_center_llh(product):
    lat = _to_numpy(getattr(product, "geometry_latitude", None))
    lon = _to_numpy(getattr(product, "geometry_longitude", None))
    height = _to_numpy(getattr(product, "geometry_height", None))

    if lat is None or lon is None:
        raise RuntimeError("geometry_latitude/geometry_longitude are not available")

    lat_arr = np.asarray(lat, dtype=float)
    lon_arr = np.asarray(lon, dtype=float)

    if lat_arr.ndim >= 2 and lon_arr.ndim >= 2:
        row = lat_arr.shape[0] // 2
        col = lat_arr.shape[1] // 2
        lat_c = float(lat_arr[row, col])
        lon_c = float(lon_arr[row, col])
        if height is not None:
            height_arr = np.asarray(height, dtype=float)
            if height_arr.ndim >= 2 and height_arr.shape[:2] == lat_arr.shape[:2]:
                h_c = float(height_arr[row, col])
            else:
                h_c = float(np.nanmedian(height_arr))
        else:
            h_c = 0.0
    else:
        lat_c = float(np.nanmedian(lat_arr))
        lon_c = float(np.nanmedian(lon_arr))
        h_c = float(np.nanmedian(np.asarray(height, dtype=float))) if height is not None else 0.0

    if not np.isfinite(h_c):
        h_c = 0.0

    return lat_c, lon_c, h_c

def _llh_to_ecef(lat_deg, lon_deg, h_m):
    lat = np.deg2rad(float(lat_deg))
    lon = np.deg2rad(float(lon_deg))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)
    n = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)

    x = (n + h_m) * cos_lat * cos_lon
    y = (n + h_m) * cos_lat * sin_lon
    z = ((1.0 - WGS84_E2) * n + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)

def _extract_scene_center_state_vectors(product, orb_xml_path=None, ref_time_override=None):
    position_sets = [
        ("orbitPositionX", "orbitPositionY", "orbitPositionZ"),
        ("sensorPositionX", "sensorPositionY", "sensorPositionZ"),
        ("satellitePositionX", "satellitePositionY", "satellitePositionZ"),
        ("spacecraftPositionX", "spacecraftPositionY", "spacecraftPositionZ"),
        ("positionX", "positionY", "positionZ"),
    ]
    velocity_sets = [
        ("orbitVelocityX", "orbitVelocityY", "orbitVelocityZ"),
        ("sensorVelocityX", "sensorVelocityY", "sensorVelocityZ"),
        ("satelliteVelocityX", "satelliteVelocityY", "satelliteVelocityZ"),
        ("spacecraftVelocityX", "spacecraftVelocityY", "spacecraftVelocityZ"),
        ("velocityX", "velocityY", "velocityZ"),
    ]
    direct_position_names = ["orbitPosition", "sensorPosition", "satellitePosition", "spacecraftPosition"]
    direct_velocity_names = ["orbitVelocity", "sensorVelocity", "satelliteVelocity", "spacecraftVelocity"]

    position, position_source = _vector_from_vector_names(product, direct_position_names)
    if position is None:
        position, position_source = _vector_from_component_names(product, position_sets)

    velocity, velocity_source = _vector_from_vector_names(product, direct_velocity_names)
    if velocity is None:
        velocity, velocity_source = _vector_from_component_names(product, velocity_sets)

    if position is None or velocity is None:
        xml_pos, xml_vel, xml_src = _state_vector_from_orbit_xml(
            product, orb_xml_path=orb_xml_path, ref_time_override=ref_time_override
        )
        if position is None and xml_pos is not None:
            position = xml_pos
            position_source = xml_src
        if velocity is None and xml_vel is not None:
            velocity = xml_vel
            velocity_source = xml_src

    return position, position_source, velocity, velocity_source

def estimate_bperp_from_products(p_m, p_s, master_root=None, secondary_root=None):
    """Estimate B_perp from two ALREADY-LOADED product objects (e.g. your
    existing BiomassProductSTA instances) using ATBD scene-center geometry.

    master_root/secondary_root: pass the product's root folder path (e.g.
    the same `master_sta`/`secondary_sta` Path you used to instantiate
    BiomassProductSTA) if you have it. When given, the real orbit-XML path
    is derived directly via _orb_xml_path_from_root() -- verified against
    an actual product folder -- instead of relying on the product object
    exposing an 'annotation_orb_file' attribute, which it may not.

    This is the only real change from the original estimate_bperp_from_scs_pair:
    it takes product objects directly instead of file paths, so it no longer
    assumes SCS. Everything downstream (_scene_center_llh,
    _extract_scene_center_state_vectors, _state_vector_from_orbit_xml) was
    already written generically against a `product` object via getattr() --
    no SCS-specific behaviour was ever in those helpers.

    Raises RuntimeError (loudly) rather than silently returning None/NaN if
    it can't find what it needs -- by design, so a missing field is visible
    immediately rather than propagating into a silently-wrong depth map.
    """
    lat_c, lon_c, h_c = _scene_center_llh(p_m)
    target_ecef = _llh_to_ecef(lat_c, lon_c, h_c)

    orb_xml_m = _orb_xml_path_from_root(master_root) if master_root is not None else None
    orb_xml_s = _orb_xml_path_from_root(secondary_root) if secondary_root is not None else None
    ref_time_m = _reference_time_from_root_name(master_root) if master_root is not None else None
    ref_time_s = _reference_time_from_root_name(secondary_root) if secondary_root is not None else None

    s1, s1_source, v1, v1_source = _extract_scene_center_state_vectors(
        p_m, orb_xml_path=orb_xml_m, ref_time_override=ref_time_m
    )
    s2, s2_source, _, _ = _extract_scene_center_state_vectors(
        p_s, orb_xml_path=orb_xml_s, ref_time_override=ref_time_s
    )

    if s1 is None or s2 is None:
        raise RuntimeError("Missing orbit position vectors for ATBD baseline estimate")
    if v1 is None:
        raise RuntimeError("Missing orbit velocity vector for ATBD baseline estimate")

    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    v1 = np.asarray(v1, dtype=float)

    baseline = s2 - s1
    los = target_ecef - s1
    los_norm = float(np.linalg.norm(los))
    vel_norm = float(np.linalg.norm(v1))

    if los_norm < 1e-9 or vel_norm < 1e-9:
        raise RuntimeError("Degenerate LOS or velocity vector for ATBD baseline estimate")

    los_hat = los / los_norm
    t_hat = v1 / vel_norm
    n_hat = np.cross(t_hat, los_hat)
    n_norm = float(np.linalg.norm(n_hat))
    if n_norm < 1e-9:
        raise RuntimeError("Could not construct a stable normal vector from track and LOS")
    n_hat = n_hat / n_norm

    b_perp = float(np.dot(baseline, n_hat))

    return {
        "b_perp_m": b_perp,
        "b_perp_source": f"ATBD_scene_center_projection({s1_source},{s2_source},{v1_source})",
        "r_slant_m": los_norm,
        "r_slant_source": "scene_center_target_range",
        "scene_center_lat": lat_c,
        "scene_center_lon": lon_c,
        "scene_center_height_m": h_c,
        "master_position_ecef": s1,
        "secondary_position_ecef": s2,
        "master_velocity_ecef": v1,
        "los_unit_vector": los_hat,
        "track_unit_vector": t_hat,
        "normal_unit_vector": n_hat,
        "orbit_master": int(getattr(p_m, "orbitNumber", -1)) if hasattr(p_m, "orbitNumber") else None,
        "orbit_secondary": int(getattr(p_s, "orbitNumber", -1)) if hasattr(p_s, "orbitNumber") else None,
    }

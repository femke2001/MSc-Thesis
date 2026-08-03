from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal
import numpy as np

PhaseMode = Literal["surface", "iem"]
DielectricMode = Literal["constant", "topp", "crim", "dobson_simple"]

C0 = 299_792_458.0
EPS0 = 8.854187817e-12

# ---------------------------------------------------------------------------
# Bir Safsaf permittivity constants (Paillou et al. 2003, Table I)
# Measured on soil samples at 500–900 MHz, south-central Egypt.
# ---------------------------------------------------------------------------

#: Modern Sand Sheet — thin surface layer, bimodal fine sand
EPS_MSS = 2.85 - 0.05j

#: Compacted Sand Layer — bimodal fine sand and granules, ~30 cm thick
EPS_CSL = 2.95 - 0.08j

#: Small Pebble Alluvium — small pebbles, clays, coarse sand, 50 cm–2 m
EPS_SPA = 3.15 - 0.15j

#: Calcified Pebble-Gravel — CaCO3 nodules, coarse alluvium, sandstone
#: This is the diagnostic paleochannel marker layer at Bir Safsaf.
EPS_CPG = 3.80 - 0.17j

#: Nubia Sandstone Bedrock — sandstone with interbedded shales, ≥1 m thick
EPS_NSB = 3.55 - 0.16j


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LayerConfig:
    """One subsurface interface/layer contribution.

    depth_m is the layer's *own* depth parameter.  The physical burial depth
    of the reflecting interface is:
        reflector_depth_m = surface_depth_m + depth_m * M
    where surface_depth_m = background_depth_m + channel_depth_amplitude_m * M
    and M is the channel membership map.

    epsilon_background  permittivity of this layer outside the channel
    epsilon_channel     permittivity of this layer inside the channel
    depth_m             depth of this layer below the surface geometry
    amplitude           dimensionless field amplitude scaling factor;
                        multiplied by M when scale_by_membership=True
    scale_by_membership True  → layer is present only inside the channel
                        False → layer is present everywhere (e.g. CSL)
    depth_follows_channel
                        True  → total depth = surface_depth_m + depth_m * M
                        False → total depth = depth_m (fixed everywhere)
    """
    epsilon_background: complex = EPS_MSS
    epsilon_channel: complex = EPS_CPG
    depth_m: float = 1.5
    amplitude: float = 0.25
    scale_by_membership: bool = True
    depth_follows_channel: bool = True


@dataclass(frozen=True)
class PaleochannelAdvancedConfig:
    """Configuration for the advanced BIOMASS-like paleochannel model.

    Default scene: Bir Safsaf single-CPG configuration.
    The surface is parameterised with the MSS permittivity (2.85 − 0.05j),
    and the primary buried reflector uses the CPG permittivity (3.80 − 0.17j),
    both from Paillou et al. (2003), Table I, at 500–900 MHz.
    The CPG reflector depth of 1.5 m represents the midpoint of the
    stratigraphically implied depth range (1.0–2.5 m) at Bir Safsaf.
    """

    # Grid
    nx: int = 240
    ny: int = 120
    dx: float = 10.0
    dy: float = 10.0

    # BIOMASS-like radar geometry
    freq_hz: float = 435e6
    theta_deg: float = 25.5
    theta_min_deg: float | None = 23.0
    theta_max_deg: float | None = 28.0
    b_perp_m: float = 800.0
    slant_range_m: float = 734_000.0

    # Channel geometry
    channel_half_width_m: float = 140.0
    channel_center_y0_m: float = 560.0
    channel_sinu_amp_m: float = 120.0
    channel_sinu_period_m: float = 1400.0

    # Surface roughness
    # MSS at Bir Safsaf is smooth bimodal fine sand; channel fill (SPA/CPG) is coarser.
    background_roughness_rms_m: float = 0.010
    channel_roughness_rms_m: float = 0.035
    correlation_length_m: float = 0.50
    roughness_spectrum: Literal["gaussian", "exponential"] = "gaussian"
    mode: PhaseMode = "iem"

    # Dielectric scene control
    # Default: constant mode using Bir Safsaf measured values (Paillou et al. 2003).
    dielectric_mode: DielectricMode = "constant"

    # Surface (background) permittivity: MSS measured value
    background_epsilon_complex: complex = EPS_MSS   # 2.85 − 0.05j

    # Channel surface permittivity: MSS also covers the channel surface
    channel_epsilon_complex: complex = EPS_MSS      # 2.85 − 0.05j

    # Moisture-based dielectric parameters (used when dielectric_mode != "constant")
    background_moisture: float = 0.01   # hyperarid MSS, <1% typical
    channel_moisture: float = 0.05      # slightly elevated in paleochannel fill
    sand_pct: float = 85.0
    clay_pct: float = 5.0
    bulk_density_g_cm3: float = 1.3
    particle_density_g_cm3: float = 2.65
    eps_solid: float = 3.0
    temp_c: float = 25.0
    salinity_ppt: float = 0.0

    # Burial depth geometry
    # surface_depth_m = background_depth_m + channel_depth_amplitude_m * M
    # reflector_depth_m = surface_depth_m + layers[0].depth_m * M
    background_depth_m: float = 0.0
    channel_depth_amplitude_m: float = 0.10   # small geometric corrugation

    # Default layer stack: single CPG reflector at 1.5 m (Bir Safsaf single-CPG config).
    # The CPG (3.80 − 0.17j) is the diagnostic paleochannel marker identified
    # by GPR at Bir Safsaf (Paillou et al. 2003; Grandjean et al. 2001).
    # Outside the channel, the background is SPA (3.15 − 0.15j), which underlies
    # both channel and inter-channel areas at this depth.
    layers: tuple[LayerConfig, ...] = (
        LayerConfig(
            epsilon_background=EPS_SPA,    # 3.15 − 0.15j  inter-channel at depth
            epsilon_channel=EPS_CPG,       # 3.80 − 0.17j  paleochannel marker
            depth_m=1.5,                   # midpoint of 1.0–2.5 m range
            amplitude=0.28,
            scale_by_membership=True,
            depth_follows_channel=True,
        ),
    )

    # Phase convention
    phase_shift_hhvv_rad: float = 0.0

    # Synthetic coherence controls
    coherence_window_px: int = 7
    rng_seed: int = 42
    add_speckle: bool = True
    amp_noise_std_background: float = 0.04
    amp_noise_std_channel: float = 0.12
    phase_noise_std_background_rad: float = 0.08
    phase_noise_std_channel_rad: float = 0.35
    temporal_coherence_background: float = 0.92   # slightly higher for dry stable sand
    temporal_coherence_channel: float = 0.72
    snr_db: float = 12.0
    volume_decorrelation_strength: float = 1.0


# ---------------------------------------------------------------------------
# Named Bir Safsaf scene configurations
# ---------------------------------------------------------------------------

def bir_safsaf_single_cpg(
    cpg_depth_m: float = 1.5,
    cpg_amplitude: float = 0.28,
    **overrides,
) -> PaleochannelAdvancedConfig:
    """Bir Safsaf single-CPG configuration (default scene).

    One CPG layer at depth cpg_depth_m below the MSS surface.
    This is the simplest physically grounded Bir Safsaf scene and matches
    the single-reflector assumption of the depth-inversion pipeline.

    The CPG depth of 1.5 m is the midpoint of the stratigraphically implied
    range:
        MSS  ~5 cm
        CSL  ~30 cm  (midpoint of 5 cm–1 m)
        SPA  ~1.15 m (midpoint of 50 cm–2 m)
        ------
        Total to CPG top: ~1.5 m

    References: Paillou et al. (2003) Table I; Grandjean et al. (2001).
    """
    return replace(
        PaleochannelAdvancedConfig(
            layers=(
                LayerConfig(
                    epsilon_background=EPS_SPA,
                    epsilon_channel=EPS_CPG,
                    depth_m=cpg_depth_m,
                    amplitude=cpg_amplitude,
                    scale_by_membership=True,
                    depth_follows_channel=True,
                ),
            ),
        ),
        **overrides,
    )


def bir_safsaf_multilayer(**overrides) -> PaleochannelAdvancedConfig:
    """Bir Safsaf full multilayer configuration.

    Three separate LayerConfig entries representing the CSL, SPA, and CPG
    layers of the Paillou et al. (2003) stratigraphy.  The MSS is handled
    as the surface dielectric (background_epsilon_complex = EPS_MSS).

    Layer stack (depths are cumulative to each interface):
        CSL top: ~0.05 m  (below 5 cm MSS)
        SPA top: ~0.35 m  (below 5 cm MSS + 30 cm CSL)
        CPG top: ~1.50 m  (below MSS + CSL + 1.15 m SPA midpoint)
        NSB     excluded — at depth ≥ 2 m, contribution negligible at P-band

    The CSL and SPA layers are present both inside and outside the channel
    (scale_by_membership=False); the CPG is present only inside the channel
    (scale_by_membership=True), consistent with GPR observations at Bir Safsaf.

    References: Paillou et al. (2003) Table I; Grandjean et al. (2001).
    """
    layers = (
        # CSL: compacted sand layer, present everywhere, ~30 cm thick
        LayerConfig(
            epsilon_background=EPS_CSL,
            epsilon_channel=EPS_CSL,
            depth_m=0.05,           # top of CSL, just below the 5 cm MSS
            amplitude=0.06,
            scale_by_membership=False,
            depth_follows_channel=False,
        ),
        # SPA: small pebble alluvium, present everywhere, 50 cm–2 m thick
        LayerConfig(
            epsilon_background=EPS_SPA,
            epsilon_channel=EPS_SPA,
            depth_m=0.35,           # top of SPA (below MSS + CSL)
            amplitude=0.10,
            scale_by_membership=False,
            depth_follows_channel=False,
        ),
        # CPG: calcified pebble-gravel, present only inside paleochannel
        LayerConfig(
            epsilon_background=EPS_SPA,   # outside channel: SPA continues
            epsilon_channel=EPS_CPG,      # inside channel: CPG diagnostic layer
            depth_m=1.5,
            amplitude=0.28,
            scale_by_membership=True,
            depth_follows_channel=True,
        ),
    )
    return replace(PaleochannelAdvancedConfig(layers=layers), **overrides)


def bir_safsaf_nsb(**overrides) -> PaleochannelAdvancedConfig:
    """Bir Safsaf two-reflector configuration: CPG + NSB bedrock.

    Adds a Nubia sandstone bedrock (NSB) reflector at ~2.5 m below the surface,
    representing the base of the paleochannel fill sequence.  Useful for testing
    whether the model can separate two close reflectors and for deep-penetration
    sensitivity analysis.

    References: Paillou et al. (2003) Table I.
    """
    layers = (
        LayerConfig(
            epsilon_background=EPS_SPA,
            epsilon_channel=EPS_CPG,
            depth_m=1.5,
            amplitude=0.28,
            scale_by_membership=True,
            depth_follows_channel=True,
        ),
        LayerConfig(
            epsilon_background=EPS_NSB,
            epsilon_channel=EPS_NSB,
            depth_m=2.5,
            amplitude=0.12,
            scale_by_membership=False,
            depth_follows_channel=False,
        ),
    )
    return replace(PaleochannelAdvancedConfig(layers=layers), **overrides)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PaleochannelAdvancedResult:
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    incidence_angle_deg: np.ndarray
    channel_membership: np.ndarray

    # Depth fields
    surface_depth_m: np.ndarray      # background_depth_m + channel_depth_amplitude_m * M
    reflector_depth_m: np.ndarray    # surface_depth_m + layers[0].depth_m * M  (physical)
    depth_map_m: np.ndarray          # alias for surface_depth_m (backward compat)

    moisture_map: np.ndarray
    epsilon_complex: np.ndarray      # surface dielectric (MSS by default)
    roughness_rms_m: np.ndarray
    correlation_length_m: np.ndarray
    ks: np.ndarray
    R_hh: np.ndarray
    R_vv: np.ndarray
    E_surface_hh: np.ndarray
    E_surface_vv: np.ndarray
    E_volume_hh: np.ndarray
    E_volume_vv: np.ndarray
    E_total_hh: np.ndarray
    E_total_vv: np.ndarray
    sigma0_hh_lin: np.ndarray
    sigma0_vv_lin: np.ndarray
    sigma0_hh_db: np.ndarray
    sigma0_vv_db: np.ndarray
    phi_hhvv_rad: np.ndarray
    phi_hhvv_deg: np.ndarray
    kz_surface_rad_per_m: np.ndarray
    kz_volume_rad_per_m: np.ndarray
    alpha_np_per_m: np.ndarray
    transmitted_angle_deg: np.ndarray
    layer_phase_delay_rad: np.ndarray
    layer_attenuation: np.ndarray

    # Coherence — two distinct observables
    gamma_copol_hhvv: np.ndarray     # copolar HH×VV* coherence, single acquisition
    gamma_vv_insar: np.ndarray       # interferometric VV1×VV2* coherence
    gamma_model: np.ndarray          # analytical: gamma_temp * gamma_vol * gamma_snr

    # Backward-compat names
    gamma_hh_sim: np.ndarray
    gamma_vv_sim: np.ndarray

    E1_vv: np.ndarray
    E2_vv: np.ndarray
    profile_x_m: np.ndarray
    profile_membership: np.ndarray
    profile_phi_hhvv_deg: np.ndarray
    channel_mean_phi_deg: float
    background_mean_phi_deg: float
    phase_contrast_deg: float


# ---------------------------------------------------------------------------
# Grid and geometry helpers
# ---------------------------------------------------------------------------

def _build_grid(cfg: PaleochannelAdvancedConfig):
    x = np.arange(cfg.nx, dtype=float) * cfg.dx
    y = np.arange(cfg.ny, dtype=float) * cfg.dy
    X, Y = np.meshgrid(x, y)
    return x, y, X, Y


def _incidence_angle_map(cfg: PaleochannelAdvancedConfig, shape: tuple) -> np.ndarray:
    ny, nx = shape
    if cfg.theta_min_deg is not None and cfg.theta_max_deg is not None:
        theta_1d = np.linspace(float(cfg.theta_min_deg), float(cfg.theta_max_deg), nx)
        return np.tile(theta_1d[None, :], (ny, 1))
    return np.full(shape, float(cfg.theta_deg))


def _channel_centerline(x: np.ndarray, cfg: PaleochannelAdvancedConfig) -> np.ndarray:
    if cfg.channel_sinu_period_m <= 0:
        return np.full_like(x, cfg.channel_center_y0_m)
    return (cfg.channel_center_y0_m
            + cfg.channel_sinu_amp_m * np.sin(2.0 * np.pi * x / cfg.channel_sinu_period_m))


def _principal_complex_sqrt(value) -> np.ndarray:
    """Square root with positive real part (branch cut on negative real axis)."""
    root = np.lib.scimath.sqrt(np.asarray(value, dtype=np.complex128))
    flip = (np.real(root) < 0.0) | ((np.abs(np.real(root)) < 1e-15) & (np.imag(root) < 0.0))
    return np.where(flip, -root, root)


def _wrapped_phase_difference(phase_hh, phase_vv, shift_rad=0.0) -> np.ndarray:
    return np.angle(np.exp(1j * (phase_hh - phase_vv + float(shift_rad))))


# ---------------------------------------------------------------------------
# Dielectric models
# ---------------------------------------------------------------------------

def debye_water_permittivity(freq_hz: float, temp_c: float = 25.0,
                              salinity_ppt: float = 0.0) -> complex:
    """Debye single-relaxation water permittivity with ionic conductivity loss."""
    eps_static = 80.1 - 0.37 * (temp_c - 20.0) - 2.6 * salinity_ppt
    eps_inf = 4.9
    tau = 1.1e-11 * np.exp(0.005346 * (25.0 - temp_c))
    omega = 2.0 * np.pi * freq_hz
    wt = omega * tau
    sigma_s_m = max(0.0, 0.001 * (1.0 + 0.2 * salinity_ppt))
    eps_real = eps_inf + (eps_static - eps_inf) / (1.0 + wt ** 2)
    eps_imag = ((eps_static - eps_inf) * wt / (1.0 + wt ** 2)) + sigma_s_m / (omega * EPS0)
    return eps_real - 1j * eps_imag


def topp_dielectric_complex(mv, freq_hz: float = 435e6,
                             salinity_ppt: float = 0.0) -> np.ndarray:
    """Topp (1980) real permittivity with Lasne et al. (2008) loss-tangent proxy.

    Loss tangent anchored to Lasne et al. (2008), Table I:
        tan_delta(mv) = 0.001 + 0.012 * mv
    """
    mv = np.asarray(mv, dtype=float)
    eps_real = np.clip(3.03 + 9.3 * mv + 146.0 * mv ** 2 - 76.7 * mv ** 3, 1.05, 80.0)
    tan_delta = 0.001 + 0.012 * np.maximum(mv, 0.0)
    if salinity_ppt > 0.0:
        omega = 2.0 * np.pi * freq_hz
        sigma_ionic = 0.01 * salinity_ppt
        tan_delta = tan_delta + sigma_ionic / (np.maximum(eps_real, 1.05) * omega * EPS0)
    eps_imag = np.clip(eps_real * tan_delta, 0.0, 5.0)
    return eps_real - 1j * eps_imag


def crim_dielectric_complex(mv, cfg: PaleochannelAdvancedConfig) -> np.ndarray:
    """Complex Refractive Index Model (CRIM) after Mironov et al. (2004)."""
    mv = np.asarray(mv, dtype=float)
    phi = np.clip(1.0 - cfg.bulk_density_g_cm3 / cfg.particle_density_g_cm3, 0.2, 0.7)
    mv_eff = np.clip(mv, 0.0, phi)
    eps_w = debye_water_permittivity(cfg.freq_hz, cfg.temp_c, cfg.salinity_ppt)
    n_eff = ((1.0 - phi) * np.sqrt(cfg.eps_solid)
             + (phi - mv_eff) * np.sqrt(1.0)
             + mv_eff * _principal_complex_sqrt(eps_w))
    return n_eff ** 2


def dobson_simple_dielectric_complex(mv, cfg: PaleochannelAdvancedConfig) -> np.ndarray:
    """Dobson-style complex dielectric approximation for synthetic scenes."""
    mv = np.asarray(mv, dtype=float)
    pb = float(cfg.bulk_density_g_cm3)
    ps = float(cfg.particle_density_g_cm3)
    eps_s = np.clip((1.01 + 0.44 * pb) ** 2 - 0.062, 2.0, 10.0)
    eps_w = debye_water_permittivity(cfg.freq_hz, cfg.temp_c, cfg.salinity_ppt)
    beta_real = (127.48 - 0.519 * cfg.sand_pct - 0.152 * cfg.clay_pct) / 100.0
    beta_imag = abs((1.33797 - 0.603 * cfg.sand_pct - 0.166 * cfg.clay_pct) / 100.0)
    eps_real = 1.0 + (pb / ps) * (eps_s - 1.0) + mv * (np.real(eps_w) ** beta_real) - mv
    eps_imag = 0.02 + mv * (abs(np.imag(eps_w)) ** beta_imag)
    return np.clip(eps_real, 1.0, 40.0) - 1j * np.clip(eps_imag, 0.0, 10.0)


def dielectric_from_moisture(mv, cfg: PaleochannelAdvancedConfig) -> np.ndarray:
    if cfg.dielectric_mode == "topp":
        return topp_dielectric_complex(mv, freq_hz=cfg.freq_hz, salinity_ppt=cfg.salinity_ppt)
    if cfg.dielectric_mode == "crim":
        return crim_dielectric_complex(mv, cfg)
    if cfg.dielectric_mode == "dobson_simple":
        return dobson_simple_dielectric_complex(mv, cfg)
    raise ValueError(f"Unsupported moisture dielectric mode: {cfg.dielectric_mode!r}")


# ---------------------------------------------------------------------------
# Fresnel coefficients
# ---------------------------------------------------------------------------

def fresnel_coefficients(epsilon_complex, theta_rad):
    """Oblique-incidence Fresnel reflection coefficients R_hh, R_vv."""
    theta, epsilon = np.broadcast_arrays(
        np.asarray(theta_rad, dtype=float),
        np.asarray(epsilon_complex, dtype=np.complex128),
    )
    cos_theta = np.cos(theta)
    sin_sq = np.sin(theta) ** 2
    root = _principal_complex_sqrt(epsilon - sin_sq)
    R_hh = (cos_theta - root) / (cos_theta + root)
    R_vv = (epsilon * cos_theta - root) / (epsilon * cos_theta + root)
    return R_hh, R_vv


def fresnel_transmission_coefficients(epsilon_complex, theta_rad):
    """Oblique-incidence Fresnel field transmission coefficients T_hh, T_vv.

    T_hh = 2 cos θ_i / (cos θ_i + √(ε − sin²θ_i))          [= 1 + R_hh, identity]
    T_vv = 2 √ε cos θ_i / (ε cos θ_i + √(ε − sin²θ_i))     [≠ 1 + R_vv at oblique θ]

    The T_vv fix (v4) corrects a 5–15% bias at 25° for ε ≈ 3–4,
    propagating directly into the HH–VV phase difference.
    """
    theta, epsilon = np.broadcast_arrays(
        np.asarray(theta_rad, dtype=float),
        np.asarray(epsilon_complex, dtype=np.complex128),
    )
    cos_theta = np.cos(theta)
    sin_sq = np.sin(theta) ** 2
    root = _principal_complex_sqrt(epsilon - sin_sq)

    T_hh = (2.0 * cos_theta) / (cos_theta + root)
    T_vv = (2.0 * _principal_complex_sqrt(epsilon) * cos_theta) / (epsilon * cos_theta + root)

    floor = 0.05
    for T in (T_hh, T_vv):
        amp = np.abs(T)
        phase = np.exp(1j * np.angle(T))
        T[:] = np.where(amp < floor, floor * phase, T)

    return T_hh, T_vv


# ---------------------------------------------------------------------------
# Surface scattering (IEM-inspired)
# ---------------------------------------------------------------------------

def _surface_spectrum_factor(k0, theta, corr_len, spectrum="gaussian") -> np.ndarray:
    q = 2.0 * k0 * np.sin(theta)
    l = np.maximum(np.asarray(corr_len, dtype=float), 1e-6)
    if spectrum == "gaussian":
        return np.exp(-0.25 * (q * l) ** 2)
    if spectrum == "exponential":
        return 1.0 / (1.0 + (q * l) ** 2) ** 1.5
    raise ValueError("roughness_spectrum must be 'gaussian' or 'exponential'")


def iem_inspired_surface_fields(epsilon_complex, theta_rad, roughness_rms_m,
                                 corr_len_m, freq_hz, spectrum="gaussian"):
    """IEM-inspired first-order surface fields for HH and VV polarisations."""
    theta, epsilon, s, l = np.broadcast_arrays(
        np.asarray(theta_rad, dtype=float),
        np.asarray(epsilon_complex, dtype=np.complex128),
        np.asarray(roughness_rms_m, dtype=float),
        np.asarray(corr_len_m, dtype=float),
    )
    wavelength = C0 / freq_hz
    k0 = 2.0 * np.pi / wavelength
    ks = k0 * s
    R_hh, R_vv = fresnel_coefficients(epsilon, theta)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sin2 = sin_t ** 2
    inv_eps = 1.0 / epsilon

    f_hh = -2.0 * R_hh * cos_t
    f_vv = 2.0 * R_vv * cos_t
    F_hh = (-2.0 * sin2 * cos_t) * (4.0 * R_hh - (1.0 - inv_eps) * (1.0 + R_hh) ** 2)
    F_vv = (2.0 * sin2 * cos_t) * (
        (1.0 - (cos_t ** 2 / epsilon - sin2)) * (1.0 - R_vv) ** 2
        + (1.0 - inv_eps) * (1.0 + R_vv) ** 2
    )

    coherent_loss = np.exp(-0.5 * (ks * cos_t) ** 2)
    spectrum_factor = _surface_spectrum_factor(k0, theta, l, spectrum=spectrum)
    rough_strength = ks ** 2 * spectrum_factor

    E_hh = R_hh * coherent_loss + (f_hh + F_hh / (8.0 * np.pi ** 2)) * rough_strength
    E_vv = R_vv * coherent_loss + (f_vv + F_vv / (8.0 * np.pi ** 2)) * rough_strength

    return {
        "ks": ks, "R_hh": R_hh, "R_vv": R_vv,
        "E_surface_hh": E_hh, "E_surface_vv": E_vv,
        "coherent_loss": coherent_loss,
        "spectrum_factor": spectrum_factor,
        "rough_strength": rough_strength,
    }


# ---------------------------------------------------------------------------
# Subsurface layer field
# ---------------------------------------------------------------------------

def _layer_field(epsilon_surface, epsilon_layer, theta_rad, freq_hz, depth_m, amplitude_scale):
    """Complex field from one buried dielectric interface.

    Uses correct oblique-incidence T_hh and T_vv.
    depth_m is the total path length from the surface to the interface.
    """
    theta, eps_surface, eps_layer, depth, amp = np.broadcast_arrays(
        np.asarray(theta_rad, dtype=float),
        np.asarray(epsilon_surface, dtype=np.complex128),
        np.asarray(epsilon_layer, dtype=np.complex128),
        np.asarray(depth_m, dtype=float),
        np.asarray(amplitude_scale, dtype=float),
    )
    k0 = 2.0 * np.pi / (C0 / freq_hz)
    n_complex = _principal_complex_sqrt(eps_layer)
    n_real = np.maximum(np.real(n_complex), 1e-9)
    alpha = k0 * np.abs(np.imag(n_complex))

    sin_theta_t = np.clip(np.sin(theta) / n_real, -0.999999, 0.999999)
    cos_theta_t = np.sqrt(1.0 - sin_theta_t ** 2)

    phase_delay = 2.0 * k0 * n_real * cos_theta_t * depth
    attenuation = np.exp(-2.0 * alpha * depth / np.maximum(cos_theta_t, 1e-6))

    T_hh, T_vv = fresnel_transmission_coefficients(eps_surface, theta)
    base = amp * attenuation * np.exp(1j * phase_delay)

    return {
        "E_hh": base * T_hh,
        "E_vv": base * T_vv,
        "alpha": alpha,
        "attenuation": attenuation,
        "theta_t_deg": np.rad2deg(np.arccos(cos_theta_t)),
        "phase_delay": phase_delay,
    }


# ---------------------------------------------------------------------------
# Coherence helpers
# ---------------------------------------------------------------------------

def _local_complex_mean(z, win):
    try:
        from scipy.ndimage import uniform_filter
        return (uniform_filter(np.real(z), size=win, mode="nearest")
                + 1j * uniform_filter(np.imag(z), size=win, mode="nearest"))
    except Exception:
        return z


def local_coherence(E1, E2, win=7):
    """Estimate coherence magnitude |<E1 E2*>| / sqrt(<|E1|²><|E2|²>)."""
    win = max(int(win), 1)
    num = _local_complex_mean(E1 * np.conj(E2), win)
    p1 = np.real(_local_complex_mean(np.abs(E1) ** 2, win))
    p2 = np.real(_local_complex_mean(np.abs(E2) ** 2, win))
    return np.clip(np.abs(num / np.sqrt(np.maximum(p1 * p2, 1e-12))), 0.0, 1.0)


def _add_complex_noise(E, rng, amp_std, phase_std):
    amp = np.maximum(1.0 + rng.normal(0.0, amp_std, size=E.shape), 0.01)
    phase = rng.normal(0.0, phase_std, size=E.shape)
    return E * amp * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# Main simulation function
# ---------------------------------------------------------------------------

def simulate_paleochannel_advanced(cfg: PaleochannelAdvancedConfig) -> PaleochannelAdvancedResult:
    """Run the 2.5D forward model and return all intermediate and output fields.

    Default scene is the Bir Safsaf single-CPG configuration.
    See module docstring and bir_safsaf_* factory functions for details.
    """
    x, y, X, Y = _build_grid(cfg)
    incidence_angle_deg = _incidence_angle_map(cfg, (cfg.ny, cfg.nx))
    theta = np.deg2rad(incidence_angle_deg)

    centerline = _channel_centerline(x, cfg)
    Yc = np.tile(centerline[None, :], (cfg.ny, 1))
    M = np.exp(-0.5 * ((Y - Yc) / max(cfg.channel_half_width_m, 1e-6)) ** 2)

    surface_depth_m = cfg.background_depth_m + cfg.channel_depth_amplitude_m * M
    moisture_map = cfg.background_moisture + (cfg.channel_moisture - cfg.background_moisture) * M

    if cfg.dielectric_mode == "constant":
        eps = (cfg.background_epsilon_complex
               + (cfg.channel_epsilon_complex - cfg.background_epsilon_complex) * M)
    else:
        eps = dielectric_from_moisture(moisture_map, cfg)

    s = (cfg.background_roughness_rms_m
         + (cfg.channel_roughness_rms_m - cfg.background_roughness_rms_m) * M)
    l = np.full_like(s, float(cfg.correlation_length_m))

    surface = iem_inspired_surface_fields(
        eps, theta, s, l, cfg.freq_hz, spectrum=cfg.roughness_spectrum)
    if cfg.mode == "surface":
        E_surface_hh = surface["R_hh"]
        E_surface_vv = surface["R_vv"]
    else:
        E_surface_hh = surface["E_surface_hh"]
        E_surface_vv = surface["E_surface_vv"]

    E_volume_hh = np.zeros_like(E_surface_hh, dtype=np.complex128)
    E_volume_vv = np.zeros_like(E_surface_vv, dtype=np.complex128)
    alpha_acc = np.zeros_like(M, dtype=float)
    atten_acc = np.ones_like(M, dtype=float)
    theta_t_acc = np.zeros_like(M, dtype=float)
    phase_acc = np.zeros_like(M, dtype=float)
    primary_layer_depth_m = cfg.layers[0].depth_m if cfg.layers else 0.0

    for layer in cfg.layers:
        eps_layer = (layer.epsilon_background
                     + (layer.epsilon_channel - layer.epsilon_background) * M)
        if layer.depth_follows_channel:
            layer_total_depth = surface_depth_m + layer.depth_m * M
        else:
            layer_total_depth = np.full_like(M, layer.depth_m)

        amp = layer.amplitude * (M if layer.scale_by_membership else np.ones_like(M))
        lf = _layer_field(eps, eps_layer, theta, cfg.freq_hz, layer_total_depth, amp)
        E_volume_hh += lf["E_hh"]
        E_volume_vv += lf["E_vv"]
        alpha_acc += lf["alpha"] * np.maximum(amp, 0.0)
        atten_acc *= np.maximum(lf["attenuation"], 1e-12)
        theta_t_acc += lf["theta_t_deg"] * np.maximum(amp, 0.0)
        phase_acc += lf["phase_delay"] * np.maximum(amp, 0.0)

    reflector_depth_m = surface_depth_m + primary_layer_depth_m * M

    amp_sum = np.zeros_like(M, dtype=float)
    for layer in cfg.layers:
        amp_sum += layer.amplitude * (M if layer.scale_by_membership else np.ones_like(M))
    amp_sum = np.maximum(amp_sum, 1e-12)
    alpha_map = alpha_acc / amp_sum
    theta_t_map = theta_t_acc / amp_sum
    phase_delay_map = phase_acc / amp_sum

    E_total_hh = E_surface_hh + E_volume_hh
    E_total_vv = E_surface_vv + E_volume_vv

    sigma0_hh_lin = np.maximum(np.abs(E_total_hh) ** 2, 1e-12)
    sigma0_vv_lin = np.maximum(np.abs(E_total_vv) ** 2, 1e-12)
    sigma0_hh_db = 10.0 * np.log10(sigma0_hh_lin)
    sigma0_vv_db = 10.0 * np.log10(sigma0_vv_lin)

    wavelength = C0 / cfg.freq_hz
    kz_surf = ((4.0 * np.pi / wavelength)
               * cfg.b_perp_m / (cfg.slant_range_m * np.maximum(np.sin(theta), 1e-8)))
    eps_real = np.maximum(np.real(eps), 1.000001)
    kz_vol = kz_surf * (eps_real * np.cos(theta)) / np.sqrt(
        np.maximum(eps_real - np.sin(theta) ** 2, 1e-8))

    phi_hhvv = _wrapped_phase_difference(
        np.angle(E_total_hh), np.angle(E_total_vv), cfg.phase_shift_hhvv_rad)
    phi_hhvv_deg = np.rad2deg(phi_hhvv)

    rng = np.random.default_rng(cfg.rng_seed)
    amp_std = (cfg.amp_noise_std_background
               + (cfg.amp_noise_std_channel - cfg.amp_noise_std_background) * M)
    phase_std = (cfg.phase_noise_std_background_rad
                 + (cfg.phase_noise_std_channel_rad - cfg.phase_noise_std_background_rad) * M)

    if cfg.add_speckle:
        E1_hh_n = _add_complex_noise(E_total_hh, rng, amp_std, phase_std)
        E1_vv_n = _add_complex_noise(E_total_vv, rng, amp_std, phase_std)
    else:
        E1_hh_n = E_total_hh.copy()
        E1_vv_n = E_total_vv.copy()

    # Copolar coherence: HH × VV* within one acquisition
    gamma_copol_hhvv = local_coherence(E1_hh_n, E1_vv_n, cfg.coherence_window_px)

    # Interferometric coherence: VV master × VV secondary
    delta_phi_insar = kz_vol * reflector_depth_m
    E2_clean_vv = E_surface_vv + E_volume_vv * np.exp(1j * delta_phi_insar)
    if cfg.add_speckle:
        E2_vv_n = _add_complex_noise(E2_clean_vv, rng, amp_std, phase_std)
    else:
        E2_vv_n = E2_clean_vv.copy()

    gamma_temp = (cfg.temporal_coherence_background
                  + (cfg.temporal_coherence_channel - cfg.temporal_coherence_background) * M)
    gamma_vol = np.exp(-cfg.volume_decorrelation_strength
                       * np.abs(kz_vol) * reflector_depth_m)
    snr_lin = 10.0 ** (cfg.snr_db / 10.0)
    gamma_snr = snr_lin / (1.0 + snr_lin)
    gamma_model = np.clip(gamma_temp * gamma_vol * gamma_snr, 0.0, 1.0)

    gamma_vv_raw = local_coherence(E1_vv_n, E2_vv_n, cfg.coherence_window_px)
    gamma_vv_insar = np.clip(gamma_vv_raw * gamma_model, 0.0, 1.0)
    gamma_hh_sim = gamma_vv_insar
    gamma_vv_sim = gamma_vv_insar

    channel_mask = M >= 0.5
    background_mask = M < 0.1
    channel_mean = float(np.nanmean(phi_hhvv_deg[channel_mask]))
    background_mean = float(np.nanmean(phi_hhvv_deg[background_mask]))
    contrast = channel_mean - background_mean

    center_idx = np.clip(
        np.round(centerline / max(cfg.dy, 1e-9)).astype(int), 0, cfg.ny - 1)
    profile_phi = phi_hhvv_deg[center_idx, np.arange(cfg.nx)]
    profile_m = M[center_idx, np.arange(cfg.nx)]

    return PaleochannelAdvancedResult(
        x=x, y=y, X=X, Y=Y,
        incidence_angle_deg=incidence_angle_deg,
        channel_membership=M,
        surface_depth_m=surface_depth_m,
        reflector_depth_m=reflector_depth_m,
        depth_map_m=surface_depth_m,
        moisture_map=moisture_map,
        epsilon_complex=eps,
        roughness_rms_m=s,
        correlation_length_m=l,
        ks=surface["ks"],
        R_hh=surface["R_hh"],
        R_vv=surface["R_vv"],
        E_surface_hh=E_surface_hh,
        E_surface_vv=E_surface_vv,
        E_volume_hh=E_volume_hh,
        E_volume_vv=E_volume_vv,
        E_total_hh=E_total_hh,
        E_total_vv=E_total_vv,
        sigma0_hh_lin=sigma0_hh_lin,
        sigma0_vv_lin=sigma0_vv_lin,
        sigma0_hh_db=sigma0_hh_db,
        sigma0_vv_db=sigma0_vv_db,
        phi_hhvv_rad=phi_hhvv,
        phi_hhvv_deg=phi_hhvv_deg,
        kz_surface_rad_per_m=kz_surf,
        kz_volume_rad_per_m=kz_vol,
        alpha_np_per_m=alpha_map,
        transmitted_angle_deg=theta_t_map,
        layer_phase_delay_rad=phase_delay_map,
        layer_attenuation=atten_acc,
        gamma_copol_hhvv=gamma_copol_hhvv,
        gamma_vv_insar=gamma_vv_insar,
        gamma_model=gamma_model,
        gamma_hh_sim=gamma_hh_sim,
        gamma_vv_sim=gamma_vv_sim,
        E1_vv=E1_vv_n,
        E2_vv=E2_vv_n,
        profile_x_m=x,
        profile_membership=profile_m,
        profile_phi_hhvv_deg=profile_phi,
        channel_mean_phi_deg=channel_mean,
        background_mean_phi_deg=background_mean,
        phase_contrast_deg=contrast,
    )


def build_inversion_inputs(res: PaleochannelAdvancedResult) -> dict:
    """Return forward-model outputs for the depth-inversion pipeline."""
    return {
        "sigma0_vv": res.sigma0_vv_lin,
        "sigma0_hh": res.sigma0_hh_lin,
        "gamma_vv": res.gamma_vv_insar,
        "gamma_copol": res.gamma_copol_hhvv,
        "theta_deg": res.incidence_angle_deg,
        "ks": res.ks,
        "true_surface_depth_m": res.surface_depth_m,
        "true_reflector_depth_m": res.reflector_depth_m,
        "true_depth_m": res.reflector_depth_m,
        "true_epsilon_real": np.real(res.epsilon_complex),
        "true_epsilon_imag": -np.imag(res.epsilon_complex),
        "true_moisture": res.moisture_map,
        "true_channel_membership": res.channel_membership,
    }


# ---------------------------------------------------------------------------
# Monte Carlo uncertainty propagation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamUncertainty:
    """Gaussian perturbation for one input parameter."""
    rel_std: float | None = None
    abs_std: float | None = None
    min_value: float | None = None
    max_value: float | None = None

    def sample(self, nominal: float, rng: np.random.Generator) -> float:
        std = abs(nominal) * self.rel_std if self.rel_std is not None else self.abs_std
        value = float(rng.normal(nominal, std)) if std else float(nominal)
        if self.min_value is not None:
            value = max(value, self.min_value)
        if self.max_value is not None:
            value = min(value, self.max_value)
        return value


@dataclass(frozen=True)
class MonteCarloUncertaintyConfig:
    """Input-parameter uncertainty for a BIOMASS-like P-band Bir Safsaf acquisition.

    CPG depth uncertainty (layer_depth_m, rel_std=0.25) reflects the
    stratigraphic variability in the Paillou et al. (2003) Table I thickness
    ranges: SPA 50 cm–2 m gives roughly ±50% variability in the depth to the
    CPG top, reduced to ±25% here as a conservative prior.
    """
    background_moisture: ParamUncertainty = ParamUncertainty(rel_std=0.30, min_value=0.0, max_value=0.5)
    channel_moisture: ParamUncertainty = ParamUncertainty(rel_std=0.30, min_value=0.0, max_value=0.5)
    background_depth_m: ParamUncertainty = ParamUncertainty(abs_std=0.05, min_value=0.0)
    channel_depth_amplitude_m: ParamUncertainty = ParamUncertainty(rel_std=0.20, min_value=0.0)
    layer_depth_m: ParamUncertainty = ParamUncertainty(rel_std=0.25, min_value=0.5, max_value=3.0)
    background_roughness_rms_m: ParamUncertainty = ParamUncertainty(rel_std=0.30, min_value=0.001)
    channel_roughness_rms_m: ParamUncertainty = ParamUncertainty(rel_std=0.30, min_value=0.001)
    sand_pct: ParamUncertainty = ParamUncertainty(abs_std=5.0, min_value=0.0, max_value=100.0)
    clay_pct: ParamUncertainty = ParamUncertainty(abs_std=3.0, min_value=0.0, max_value=100.0)
    bulk_density_g_cm3: ParamUncertainty = ParamUncertainty(abs_std=0.10, min_value=0.8, max_value=2.2)
    temp_c: ParamUncertainty = ParamUncertainty(abs_std=5.0)
    b_perp_m: ParamUncertainty = ParamUncertainty(rel_std=0.10)
    slant_range_m: ParamUncertainty = ParamUncertainty(rel_std=0.01)
    snr_db: ParamUncertainty = ParamUncertainty(abs_std=2.0)


def perturb_config(
    cfg: PaleochannelAdvancedConfig,
    mc_cfg: MonteCarloUncertaintyConfig,
    rng: np.random.Generator,
) -> PaleochannelAdvancedConfig:
    """Draw one randomly perturbed copy of cfg according to mc_cfg."""
    new_layer0 = replace(cfg.layers[0],
                         depth_m=mc_cfg.layer_depth_m.sample(cfg.layers[0].depth_m, rng))
    return replace(
        cfg,
        background_moisture=mc_cfg.background_moisture.sample(cfg.background_moisture, rng),
        channel_moisture=mc_cfg.channel_moisture.sample(cfg.channel_moisture, rng),
        background_depth_m=mc_cfg.background_depth_m.sample(cfg.background_depth_m, rng),
        channel_depth_amplitude_m=mc_cfg.channel_depth_amplitude_m.sample(
            cfg.channel_depth_amplitude_m, rng),
        background_roughness_rms_m=mc_cfg.background_roughness_rms_m.sample(
            cfg.background_roughness_rms_m, rng),
        channel_roughness_rms_m=mc_cfg.channel_roughness_rms_m.sample(
            cfg.channel_roughness_rms_m, rng),
        sand_pct=mc_cfg.sand_pct.sample(cfg.sand_pct, rng),
        clay_pct=mc_cfg.clay_pct.sample(cfg.clay_pct, rng),
        bulk_density_g_cm3=mc_cfg.bulk_density_g_cm3.sample(cfg.bulk_density_g_cm3, rng),
        temp_c=mc_cfg.temp_c.sample(cfg.temp_c, rng),
        b_perp_m=mc_cfg.b_perp_m.sample(cfg.b_perp_m, rng),
        slant_range_m=mc_cfg.slant_range_m.sample(cfg.slant_range_m, rng),
        snr_db=mc_cfg.snr_db.sample(cfg.snr_db, rng),
        layers=(new_layer0,) + cfg.layers[1:],
        rng_seed=int(rng.integers(0, 2 ** 31 - 1)),
    )


@dataclass(frozen=True)
class MonteCarloUncertaintyResult:
    n_realizations: int
    eps_real_mean: np.ndarray
    eps_real_std: np.ndarray
    eps_real_p05: np.ndarray
    eps_real_p95: np.ndarray
    eps_imag_mean: np.ndarray
    eps_imag_std: np.ndarray
    eps_imag_p05: np.ndarray
    eps_imag_p95: np.ndarray
    mv_mean: np.ndarray
    mv_std: np.ndarray
    mv_p05: np.ndarray
    mv_p95: np.ndarray
    reflector_depth_mean: np.ndarray
    reflector_depth_std: np.ndarray
    reflector_depth_p05: np.ndarray
    reflector_depth_p95: np.ndarray
    channel_membership: np.ndarray
    x: np.ndarray
    y: np.ndarray
    nominal_eps_real: np.ndarray
    nominal_eps_imag: np.ndarray
    nominal_mv: np.ndarray
    nominal_reflector_depth: np.ndarray
    failed_realizations: int


def monte_carlo_uncertainty(
    base_cfg: PaleochannelAdvancedConfig,
    mc_cfg: MonteCarloUncertaintyConfig | None = None,
    n_realizations: int = 300,
    rng_seed: int = 0,
) -> MonteCarloUncertaintyResult:
    """Propagate input-parameter uncertainty through simulate_paleochannel_advanced.

    Reports per-pixel spread in epsilon, moisture, and reflector depth.
    The layer_depth_m uncertainty is bounded to [0.5, 3.0] m reflecting the
    stratigraphic variability in the Bir Safsaf CPG depth range.
    """
    if mc_cfg is None:
        mc_cfg = MonteCarloUncertaintyConfig()

    rng = np.random.default_rng(rng_seed)
    nominal_res = simulate_paleochannel_advanced(base_cfg)
    shape = nominal_res.channel_membership.shape

    eps_real_stack = np.full((n_realizations,) + shape, np.nan)
    eps_imag_stack = np.full((n_realizations,) + shape, np.nan)
    mv_stack = np.full((n_realizations,) + shape, np.nan)
    depth_stack = np.full((n_realizations,) + shape, np.nan)

    failed = 0
    for i in range(n_realizations):
        try:
            cfg_i = perturb_config(base_cfg, mc_cfg, rng)
            res_i = simulate_paleochannel_advanced(cfg_i)
            eps_real_stack[i] = np.real(res_i.epsilon_complex)
            eps_imag_stack[i] = -np.imag(res_i.epsilon_complex)
            mv_stack[i] = res_i.moisture_map
            depth_stack[i] = res_i.reflector_depth_m
        except Exception:
            failed += 1

    def stats(stack):
        return (
            np.nanmean(stack, axis=0),
            np.nanstd(stack, axis=0),
            np.nanpercentile(stack, 5, axis=0),
            np.nanpercentile(stack, 95, axis=0),
        )

    er_m, er_s, er_p05, er_p95 = stats(eps_real_stack)
    ei_m, ei_s, ei_p05, ei_p95 = stats(eps_imag_stack)
    mv_m, mv_s, mv_p05, mv_p95 = stats(mv_stack)
    d_m, d_s, d_p05, d_p95 = stats(depth_stack)

    return MonteCarloUncertaintyResult(
        n_realizations=n_realizations,
        eps_real_mean=er_m, eps_real_std=er_s, eps_real_p05=er_p05, eps_real_p95=er_p95,
        eps_imag_mean=ei_m, eps_imag_std=ei_s, eps_imag_p05=ei_p05, eps_imag_p95=ei_p95,
        mv_mean=mv_m, mv_std=mv_s, mv_p05=mv_p05, mv_p95=mv_p95,
        reflector_depth_mean=d_m, reflector_depth_std=d_s,
        reflector_depth_p05=d_p05, reflector_depth_p95=d_p95,
        channel_membership=nominal_res.channel_membership,
        x=nominal_res.x, y=nominal_res.y,
        nominal_eps_real=np.real(nominal_res.epsilon_complex),
        nominal_eps_imag=-np.imag(nominal_res.epsilon_complex),
        nominal_mv=nominal_res.moisture_map,
        nominal_reflector_depth=nominal_res.reflector_depth_m,
        failed_realizations=failed,
    )

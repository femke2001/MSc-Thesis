
import numpy as np


def debye_water_permittivity(freq_hz, temp_c=20.0, salinity_ppt=0.0):
    """
    Calculate complex permittivity of water using modified Debye relaxation.
    
    Parameters:
    -----------
    freq_hz : float or array
        Frequency in Hz
    temp_c : float
        Temperature in Celsius (default 20°C)
    salinity_ppt : float
        Salinity in parts per thousand (default 0.0 ppt)
    
    Returns:
    --------
    eps_fw : complex or array of complex
        Complex relative permittivity of water
        
    References:
    -----------
    Stogryn (1986) empirical model for water dielectric properties
    Lane & Saxton (1952) for Debye relaxation with ionic conductivity
    """
    # Static permittivity (Stogryn empirical, temp/salinity corrected)
    # Simplified: eps_w0 ≈ 80.1 at 20°C, 0 ppt
    eps_w0 = 80.1 - 0.37 * (temp_c - 20.0) - 2.6 * salinity_ppt
    
    # High-frequency limit
    eps_w_inf = 4.9  # essentially independent of T, S
    
    # Relaxation time (modified Debye)
    # tau_w ≈ 1.1e-11 at 20°C
    tau_w = 1.1e-11 * np.exp(0.005346 * (25.0 - temp_c))
    
    # Ionic conductivity (simplified for fresh water / low salinity)
    sigma_mv = 1.0 + 0.2 * salinity_ppt  # mS/m, very approximate
    
    # Relative permittivity of free space
    eps_0 = 8.854e-12  # F/m
    
    # Real part
    eps_fw_real = eps_w_inf + (eps_w0 - eps_w_inf) / (1.0 + (2.0 * np.pi * freq_hz * tau_w) ** 2)
    
    # Imaginary part (includes conductivity loss)
    eps_fw_imag = (eps_w0 - eps_w_inf) * (2.0 * np.pi * freq_hz * tau_w) / (
        1.0 + (2.0 * np.pi * freq_hz * tau_w) ** 2
    ) + sigma_mv / (2.0 * np.pi * freq_hz * eps_0)
    
    return eps_fw_real + 1j * eps_fw_imag


def dobson_semiempirical_dielectric(
    mv,
    sand_pct=50.0,
    clay_pct=20.0,
    pb=1.3,
    ps=2.65,
    freq_hz=1.4e9,
    temp_c=20.0,
    salinity_ppt=0.0,
):
    """
    Calculate soil dielectric permittivity using Dobson et al. (1985) semiempirical mixing model.
    
    This model uses refractive mixing (Birchak approach) with texture-dependent coefficients
    and explicitly accounts for bound water and bulk water fractions.
    
    Parameters:
    -----------
    mv : float or array
        Volumetric moisture content (cm³/cm³, range 0-0.5)
    sand_pct : float or array
        Sand percentage by weight (0-100)
    clay_pct : float or array
        Clay percentage by weight (0-100)
    pb : float
        Bulk density (g/cm³, typically 1.0-1.7)
    ps : float
        Specific density of soil solids (g/cm³, typically 2.6-2.7)
    freq_hz : float
        Frequency in Hz (default 1.4 GHz for L-band)
    temp_c : float
        Temperature in Celsius (default 20°C)
    salinity_ppt : float
        Salinity in ppt (default 0, fresh water)
    
    Returns:
    --------
    eps : complex or array of complex
        Complex relative permittivity of soil-water mixture
    
    References:
    -----------
    Dobson, M. C., et al. (1985). "Microwave dielectric behavior of wet soil-Part II: 
    Dielectric mixing models." IEEE Transactions on Geoscience and Remote Sensing, 
    GE-23(1), 35-46. DOI: 10.1109/TGRS.1985.289498
    
    Equations:
    ----------
    eps_m = 1 + (pb/ps)(eps_s - 1) + m_v * eps_fw^beta - m_v
    
    Where:
    beta' = (127.48 - 0.519*S - 0.152*C) / 100
    beta'' = (1.33797 - 0.603*S - 0.166*C) / 100
    sigma_eff = 1.645 + 1.939*pb - 0.02013*S + 0.01594*C
    """
    
    # Ensure arrays
    mv = np.asarray(mv, dtype=float)
    sand_pct = np.asarray(sand_pct, dtype=float)
    clay_pct = np.asarray(clay_pct, dtype=float)
    pb = np.asarray(pb, dtype=float)
    ps = np.asarray(ps, dtype=float)
    
    # Dry soil permittivity (empirical fit)
    # eps_s ≈ (1.01 + 0.44*pb)^2 - 0.062
    eps_s = (1.01 + 0.44 * pb) ** 2 - 0.062
    eps_s = np.clip(eps_s, 2.0, 10.0)  # Physical bounds
    
    # Free water permittivity (complex, frequency/temp/salinity dependent)
    eps_fw = debye_water_permittivity(freq_hz, temp_c, salinity_ppt)
    
    # Texture-dependent mixing coefficients (Equations 30-31)
    beta_real = (127.48 - 0.519 * sand_pct - 0.152 * clay_pct) / 100.0
    beta_imag = (1.33797 - 0.603 * sand_pct - 0.166 * clay_pct) / 100.0
    
    # Shape factor (refractive mixing, a=0.65 optimized in paper)
    a = 0.65
    
    # Semiempirical mixing formula (Equation 28-29, generalized form)
    # eps_m = 1 + (pb/ps)(eps_s - 1) + m_v * eps_fw^beta - m_v
    
    # Real part
    eps_m_real = 1.0 + (pb / ps) * (eps_s - 1.0) + mv * np.real(eps_fw) ** beta_real - mv
    
    # Imaginary part (conductivity contribution, simplified)
    sigma_eff = 1.645 + 1.939 * pb - 0.02013 * sand_pct + 0.01594 * clay_pct
    eps_0 = 8.854e-12  # F/m
    sigma_loss = sigma_eff / (2.0 * np.pi * freq_hz * eps_0)
    
    eps_m_imag = mv * np.imag(eps_fw) ** beta_imag 
    
    eps_m = eps_m_real + 1j * eps_m_imag
    
    # Physical bounds
    eps_m_real = np.clip(np.real(eps_m), 1.0, 40.0)
    eps_m_imag = np.clip(np.imag(eps_m), 0.0, 20.0)
    
    return eps_m_real + 1j * eps_m_imag


def dobson_moisture_to_dielectric(
    mv,
    sand_pct=50.0,
    clay_pct=20.0,
    pb=1.3,
    ps=2.65,
    freq_ghz=1.4,
    temp_c=20.0,
):
    """
    Convenience wrapper: convert moisture to dielectric using Dobson model.
    
    Parameters as above, but frequency in GHz.
    """
    return dobson_semiempirical_dielectric(
        mv=mv,
        sand_pct=sand_pct,
        clay_pct=clay_pct,
        pb=pb,
        ps=ps,
        freq_hz=freq_ghz * 1e9,
        temp_c=temp_c,
    )

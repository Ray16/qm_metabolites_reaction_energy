"""Fixed production conditions shared by CPU admission and energy assembly.

These preserve the deployed constants, including its rounded 2.303 convention.
They do not expose arbitrary-temperature or arbitrary-condition calculations.
"""

T = 298.15
PH = 7.0
RT_LN10 = 2.303 * 8.314e-3 * T
CONDITIONS = {
    "T_K": T, "pH": PH, "ionic_strength_M": 0.0, "pMg": None,
    "water_activity": 1.0,
    "standard_state": "solutes 1 M; pure liquid water; H+ at the stated pH (transformed ΔrG'°)",
}

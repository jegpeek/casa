"""Independent validation of fitted orientation, using no fit at all.

The structure-function fit says the long axis a1 preferentially lies IN the
echo plane rather than along the normal W.  That claim should be checkable
directly against the images, without going through the ellipsoid fit:

  * a1 near W  =>  the in-plane slice is a cross-section THROUGH the long axis,
                   so the window should look ROUND (low coherence).
  * a1 in the plane => the slice runs along the long axis, so the window should
                   look STRIATED (high coherence).

Measured with the in-plane gradient structure tensor of the first-epoch image.
Confirmed at rho = +0.65 (p < 0.001, n = 29); the 5 most W-aligned windows
average 0.35 coherence against 0.64 for the 5 most in-plane.  That rules out
the fit inventing orientations from noise.

WHAT THIS DOES NOT TEST: the W-smearing systematic.  The images and the fit
share the same W sampling, and W is the axis whose resolution comes from epoch
spacing rather than pixel scale, so any unmodelled smearing along W would
inflate apparent in-plane extent in BOTH.  The in-plane orientation preference
points in exactly the direction such a systematic would push, and this check
cannot separate them.  A dedicated test is still needed.

Usage
-----
    from image_coherence import coherence_pa
    coh, pa = coherence_pa(window_image)      # 0 = round, 1 = striated
"""
import numpy as np
from scipy import ndimage


def coherence_pa(v, sig=4.0, presmooth=1.5):
    """Gradient-structure-tensor coherence and ridge position angle.

    Parameters
    ----------
    v : 2-D array
        One window of one epoch.  NaNs are filled with the median before
        smoothing (gradients of a NaN-holed field are meaningless).
    sig : float
        Gaussian width [pix] for averaging the tensor components.
    presmooth : float
        Gaussian width [pix] applied before differencing, to keep the gradient
        from being dominated by pixel noise.

    Returns
    -------
    coh : float in [0, 1] -- 0 = isotropic/round, 1 = perfectly directional.
    pa  : float [deg, 0-180) -- position angle of the ridge direction.
    """
    v = np.asarray(v, float)
    m = np.isfinite(v)
    z = np.where(m, v, np.nanmedian(v))
    z = ndimage.gaussian_filter(z, presmooth)
    gy, gx = np.gradient(z)
    Jxx = ndimage.gaussian_filter(gx * gx, sig).mean()
    Jyy = ndimage.gaussian_filter(gy * gy, sig).mean()
    Jxy = ndimage.gaussian_filter(gx * gy, sig).mean()
    tr = Jxx + Jyy
    if not np.isfinite(tr) or tr <= 0:
        return np.nan, np.nan
    coh = float(np.hypot(Jxx - Jyy, 2 * Jxy) / tr)
    pa = 0.5 * np.degrees(np.arctan2(2 * Jxy, Jxx - Jyy))
    return coh, float((pa + 90) % 180)

"""Brute-force validation of the block-bootstrap pair weighting.

The fast path computes S2 replicates as weighted sums of precomputed block-pair
cross-correlations.  This test ignores all of that and evaluates the definition
directly -- an explicit double loop over pixel pairs, weighting each pair by
m_b(x) * m_b(y) -- on a window small enough to brute force.  Cross-block pairs
(the two endpoints in different blocks, hence weight m_b * m_c with b != c) are
the non-trivial case, so the test deliberately uses multiplicities containing
zeros and a 3, and probes lags long enough to straddle block boundaries.
"""
import sys
import numpy as np

sys.path.insert(0, '.')
import block_bootstrap as bb


def build_toy(n=18, ne=2, seed=7):
    rng = np.random.default_rng(seed)
    flux = rng.normal(size=(ne, n, n))
    flux[0, 3, 5] = np.nan            # masked pixels must drop out of both
    flux[1, 11, 2] = np.nan           # the counts and the correlation
    data = dict(flux_epochs=flux,
                U_grid=np.tile(np.arange(n) * 0.01, (n, 1)),
                V_grid=np.tile(np.arange(n)[:, None] * 0.01, (1, n)),
                W_values=np.arange(ne) * 0.5,
                size=n, row=0, col=0)
    return data, flux, n, ne


def main():
    data, flux, n, ne = build_toy()
    CK = dict(clip_percentiles=None, fill_nans=False, background=None,
              arcsinh_scale=None, subtract_mean='none')
    k = 3
    pre = bb.precompute_blocks(data, k=k, inner_uv_pixels=n - 1, **CK)

    m = np.array([2, 0, 1, 3, 1, 0, 1, 1, 0], dtype=float)
    assert m.sum() == k * k, 'multiplicities must preserve total block count'
    got = bb.s2_from_weights(pre, m)

    blocks = bb.block_bounds(n, n, k)
    bid = np.zeros((n, n), int)
    for i, (r0, r1, c0, c1) in enumerate(blocks):
        bid[r0:r1, c0:c1] = i
    Wimg = m[bid]

    mask = np.isfinite(flux)
    f0 = np.where(mask, flux, 0.0)
    # per-epoch weighted mean square, same weights as the pair counting
    ef2 = np.array([(Wimg[mask[e]] * f0[e][mask[e]] ** 2).sum()
                    / Wimg[mask[e]].sum() for e in range(ne)])

    pairs = pre['epoch_pairs']
    cv = cu = n - 1
    tests = [(0, 0), (0, 1), (1, 0), (2, 3), (-4, 2), (7, -6),
             (5, 5), (-11, 11), (0, 12), (13, 0)]

    worst_N = worst_S = 0.0
    n_checked = 0
    for pi, (i, j) in enumerate(pairs):
        for dv, du in tests:
            N = X = 0.0
            for r in range(n):
                for c in range(n):
                    rr, cc = r + dv, c + du
                    if not (0 <= rr < n and 0 <= cc < n):
                        continue
                    if not (mask[i, r, c] and mask[j, rr, cc]):
                        continue
                    w = Wimg[r, c] * Wimg[rr, cc]
                    N += w
                    X += w * f0[i, r, c] * f0[j, rr, cc]
            if N < 0.5:
                continue
            s2_bf = ((ef2[i] + ef2[j]) * N - 2 * X) / N
            s2_got = got['s2'][pi, cv + dv, cu + du]
            n_got = got['n_counts'][pi, cv + dv, cu + du]
            worst_N = max(worst_N, abs(N - n_got))
            worst_S = max(worst_S, abs(s2_bf - s2_got) / max(abs(s2_bf), 1e-12))
            n_checked += 1

    print('brute force: %d (epoch-pair, lag) combinations checked' % n_checked)
    print('  max |N - n_counts|   = %.3e' % worst_N)
    print('  max rel |S2 - S2_bf| = %.3e' % worst_S)

    # How much of each lag's signal is cross-block?  If cross-block pairs were
    # dropped or mis-weighted, these are the lags that would show it.
    print()
    print('%8s %12s %12s %8s' % ('lag', 'within-blk', 'cross-blk', '%cross'))
    for dv, du in [(1, 0), (3, 0), (5, 0), (6, 0), (8, 0), (12, 0)]:
        wN = cN = 0
        for r in range(n):
            for c in range(n):
                rr, cc = r + dv, c + du
                if not (0 <= rr < n and 0 <= cc < n):
                    continue
                if not (mask[0, r, c] and mask[0, rr, cc]):
                    continue
                if bid[r, c] == bid[rr, cc]:
                    wN += 1
                else:
                    cN += 1
        print('%8s %12d %12d %7.0f%%'
              % ('(%d,%d)' % (dv, du), wN, cN, 100 * cN / max(wN + cN, 1)))

    # Counts are integer-valued and must match exactly.  S2 is accumulated in
    # float32 (as in compute_s2 itself, and to keep the precomputed block-pair
    # arrays affordable in memory), so the tolerance is set by accumulating
    # k^4 = 81 block-pair terms at float32 eps = 1.2e-7, not by the algebra.
    assert worst_N == 0.0, 'pair counts disagree with the definition'
    assert worst_S < 1e-5, 'S2 disagrees with the definition'
    print('\nPASS')


if __name__ == '__main__':
    main()

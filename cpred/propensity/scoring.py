"""Propensity scoring formula (Lo et al. 2012, eq 1.1-1.3).

Sp(i) = W(i) * ((fe(i) + fmin(i)) / (fc(i) + fmin(i)) - 1)

where:
  fe(i)   = frequency of element i in experimental CP sites
  fc(i)   = frequency of element i in comparison set (whole proteins)
  fmin(i) = (1/ne + 1/nc) / 2  (smoothing term)
  W(i)    = 1 - p(i)           (statistical weight)
  ne      = total number of experimental elements
  nc      = total number of comparison elements
  p(i)    = p-value from permutation test

Elements can be: single AA, di-residue, oligo-residue, DSSP state,
Ramachandran code, or kappa-alpha code.

GPU acceleration: when a CUDA GPU is available, the permutation test for
all elements is batched onto the GPU, reducing wall time from days to minutes
for large element sets (e.g. oligo-residue triplets).
"""

from __future__ import annotations

import sys
from collections import Counter

import numpy as np

try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


def compute_frequencies(elements: list[str]) -> dict[str, float]:
    """Compute normalized frequencies of elements."""
    counts = Counter(elements)
    total = len(elements)
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def permutation_test(experimental: list[str], comparison: list[str],
                     element: str, n_permutations: int = 1000,
                     rng: np.random.Generator | None = None) -> float:
    """Compute p-value via permutation test (CPU, single element).

    Tests whether the frequency of `element` in `experimental` is
    significantly different from its frequency in `comparison`.

    Args:
        experimental: Elements from CP site windows.
        comparison: Elements from whole protein sequences.
        element: The specific element to test.
        n_permutations: Number of permutations.
        rng: Random number generator.

    Returns:
        p-value (0 to 1).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_exp = len(experimental)
    n_total = n_exp + len(comparison)

    fe = experimental.count(element) / max(n_exp, 1)
    fc = comparison.count(element) / max(len(comparison), 1)
    observed_diff = abs(fe - fc)

    combined = experimental + comparison
    count_extreme = 0

    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        perm_exp = [combined[i] for i in perm[:n_exp]]
        perm_comp = [combined[i] for i in perm[n_exp:]]
        perm_fe = perm_exp.count(element) / max(n_exp, 1)
        perm_fc = perm_comp.count(element) / max(len(perm_comp), 1)
        if abs(perm_fe - perm_fc) >= observed_diff:
            count_extreme += 1

    return count_extreme / n_permutations


def compute_propensity(fe: float, fc: float, pval: float,
                       n_exp: int = 1, n_comp: int = 1) -> float:
    """Compute propensity score for a single element (eq 1.1-1.3).

    Sp = W(i) * ((fe + fmin) / (fc + fmin) - 1)
    where fmin = (1/ne + 1/nc) / 2, W = 1 - pval

    Args:
        fe: Frequency in experimental (CP site) set.
        fc: Frequency in comparison (whole protein) set.
        pval: p-value from permutation test.
        n_exp: Total number of experimental elements.
        n_comp: Total number of comparison elements.

    Returns:
        Propensity score.
    """
    fmin = (1.0 / max(n_exp, 1) + 1.0 / max(n_comp, 1)) / 2.0
    denom = fc + fmin
    if denom < 1e-10:
        return 0.0
    return (1.0 - pval) * ((fe + fmin) / denom - 1.0)


# =========================================================================
# GPU-accelerated batch permutation test
# =========================================================================

def _gpu_available() -> bool:
    """Check if a CUDA GPU is available for acceleration."""
    return HAS_TORCH and torch.cuda.is_available()


def _batch_permutation_test_gpu(
    combined_ids: np.ndarray,
    n_exp: int,
    n_comp: int,
    element_masks: np.ndarray,
    observed_diffs: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
    element_batch_size: int = 2048,
) -> np.ndarray:
    """Run permutation test for ALL elements at once on GPU.

    Args:
        combined_ids: (n_total,) int array — integer-encoded combined list.
        n_exp: Number of experimental elements.
        n_comp: Number of comparison elements.
        element_masks: (n_elements, n_total) bool — mask[e, i] = True if
            combined[i] is element e.
        observed_diffs: (n_elements,) — observed |fe - fc| per element.
        n_permutations: Number of permutations.
        seed: Random seed.
        element_batch_size: Process elements in chunks of this size to
            limit GPU memory usage.

    Returns:
        (n_elements,) p-values.
    """
    device = torch.device("cuda")
    n_total = len(combined_ids)
    n_elements = len(observed_diffs)

    # Pre-generate all permutation indices on GPU: (n_perms, n_total)
    # Use CPU RNG to generate, then transfer (torch.randperm is single-perm)
    print(f"    GPU: generating {n_permutations} permutations of {n_total} elements...",
          flush=True)
    rng = np.random.default_rng(seed)
    perm_indices_np = np.empty((n_permutations, n_total), dtype=np.int64)
    for p in range(n_permutations):
        perm_indices_np[p] = rng.permutation(n_total)

    # We only need the first n_exp indices from each permutation
    perm_exp_indices = torch.from_numpy(
        perm_indices_np[:, :n_exp].copy()
    ).to(device)  # (n_perms, n_exp)
    del perm_indices_np

    observed_diffs_t = torch.from_numpy(observed_diffs).float().to(device)
    count_extreme = torch.zeros(n_elements, dtype=torch.int32, device=device)

    # Process elements in batches to control memory
    for batch_start in range(0, n_elements, element_batch_size):
        batch_end = min(batch_start + element_batch_size, n_elements)
        batch_size = batch_end - batch_start

        # masks for this batch: (batch_size, n_total)
        masks_batch = torch.from_numpy(
            element_masks[batch_start:batch_end].astype(np.float32)
        ).to(device)

        # Total count of each element in combined list
        total_counts = masks_batch.sum(dim=1)  # (batch_size,)

        obs_diff_batch = observed_diffs_t[batch_start:batch_end]  # (batch_size,)

        # Process permutations in sub-batches to control memory
        # Each perm needs: gather (batch_size, n_exp) from masks, sum -> (batch_size,)
        perm_batch_size = 100  # process 100 permutations at a time
        for pstart in range(0, n_permutations, perm_batch_size):
            pend = min(pstart + perm_batch_size, n_permutations)
            n_p = pend - pstart

            # perm_exp_indices[pstart:pend] is (n_p, n_exp)
            perm_batch = perm_exp_indices[pstart:pend]  # (n_p, n_exp)

            # Gather: for each element in batch, for each perm, count how many
            # of the permuted experimental indices match that element
            # masks_batch: (batch_size, n_total)
            # perm_batch:  (n_p, n_exp)
            #
            # We want: counts[e, p] = sum of masks_batch[e, perm_batch[p, :]]
            # = masks_batch[e].index_select(perm_batch[p]) . sum()
            #
            # Expand perm_batch to (n_p, n_exp) -> gather from masks
            # Efficient: masks_batch[:, perm_batch] -> (batch_size, n_p, n_exp)
            gathered = masks_batch[:, perm_batch.reshape(-1)].reshape(
                batch_size, n_p, n_exp
            )  # (batch_size, n_p, n_exp)
            exp_counts = gathered.sum(dim=2)  # (batch_size, n_p)

            # Compute fe and fc for each (element, perm)
            perm_fe = exp_counts / n_exp  # (batch_size, n_p)
            perm_fc = (total_counts.unsqueeze(1) - exp_counts) / n_comp
            perm_diff = (perm_fe - perm_fc).abs()  # (batch_size, n_p)

            # Compare to observed
            extreme = perm_diff >= obs_diff_batch.unsqueeze(1)  # (batch_size, n_p)
            count_extreme[batch_start:batch_end] += extreme.sum(dim=1).int()

        if batch_end % max(element_batch_size, 1) == 0 or batch_end == n_elements:
            print(f"    GPU: processed elements {batch_end}/{n_elements}",
                  flush=True)

    pvals = count_extreme.float() / n_permutations
    return pvals.cpu().numpy()


def _batch_permutation_test_cpu(
    experimental: list[str],
    comparison: list[str],
    all_elements: list[str],
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """CPU vectorised batch permutation test using numpy.

    Encodes all elements as integers, generates all permutations at once,
    and computes p-values for every element in a single numpy pass —
    avoiding the per-element Python loop that made this take days.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_exp = len(experimental)
    n_comp = len(comparison)
    n_total = n_exp + n_comp
    n_elements = len(all_elements)

    fe_freq = compute_frequencies(experimental)
    fc_freq = compute_frequencies(comparison)

    # Encode combined list as integer IDs
    elem_to_id = {e: i for i, e in enumerate(all_elements)}
    combined_ids = np.array(
        [elem_to_id[c] for c in experimental + comparison], dtype=np.int32
    )

    # observed |fe - fc| per element
    observed_diffs = np.array(
        [abs(fe_freq.get(e, 0.0) - fc_freq.get(e, 0.0)) for e in all_elements],
        dtype=np.float64,
    )

    # Count of each element in the combined list (for fc computation)
    total_counts = np.array(
        [(combined_ids == i).sum() for i in range(n_elements)], dtype=np.float64
    )

    # Generate all permutation indices: (n_permutations, n_total)
    # Only the first n_exp columns are needed (the "experimental" split)
    print(f"    CPU: generating {n_permutations} permutations of {n_total} elements...",
          flush=True)
    perm_indices = np.empty((n_permutations, n_exp), dtype=np.int32)
    for p in range(n_permutations):
        perm_indices[p] = rng.permutation(n_total)[:n_exp]

    # Process elements in batches to keep memory reasonable
    element_batch_size = 512
    count_extreme = np.zeros(n_elements, dtype=np.int64)

    for batch_start in range(0, n_elements, element_batch_size):
        batch_end = min(batch_start + element_batch_size, n_elements)

        # Build mask for this batch: (batch_size, n_total)
        batch_ids = np.arange(batch_start, batch_end, dtype=np.int32)
        masks = (combined_ids[np.newaxis, :] == batch_ids[:, np.newaxis])  # (batch, n_total)

        obs_diff_batch = observed_diffs[batch_start:batch_end]  # (batch,)
        total_counts_batch = total_counts[batch_start:batch_end]  # (batch,)

        # For each permutation, count experimental hits per element:
        # perm_indices: (n_perms, n_exp)
        # masks[:, perm_indices]: (batch, n_perms, n_exp) -- too large, do in perm chunks
        perm_batch_size = 200
        for pstart in range(0, n_permutations, perm_batch_size):
            pend = min(pstart + perm_batch_size, n_permutations)
            p_idx = perm_indices[pstart:pend]  # (n_p, n_exp)

            # masks[:, p_idx] -> (batch, n_p, n_exp)
            exp_counts = masks[:, p_idx].sum(axis=2).astype(np.float64)  # (batch, n_p)

            perm_fe = exp_counts / n_exp
            perm_fc = (total_counts_batch[:, np.newaxis] - exp_counts) / n_comp
            perm_diff = np.abs(perm_fe - perm_fc)  # (batch, n_p)

            count_extreme[batch_start:batch_end] += (
                perm_diff >= obs_diff_batch[:, np.newaxis]
            ).sum(axis=1)

        print(f"    CPU: processed elements {batch_end}/{n_elements}", flush=True)

    pvals_arr = count_extreme / n_permutations
    return {elem: pvals_arr[i] for i, elem in enumerate(all_elements)}


def build_propensity_table(experimental_elements: list[str],
                           comparison_elements: list[str],
                           n_permutations: int = 1000,
                           rng: np.random.Generator | None = None,
                           use_gpu: bool = True,
                           ) -> dict[str, float]:
    """Build a propensity lookup table for all observed elements.

    Automatically uses GPU acceleration when available and use_gpu=True.
    Falls back to CPU with per-element progress otherwise.

    Args:
        experimental_elements: Elements from CP site windows.
        comparison_elements: Elements from whole protein sequences.
        n_permutations: Permutations for p-value computation.
        rng: Random number generator (used for CPU path and as seed source for GPU).
        use_gpu: Whether to attempt GPU acceleration.

    Returns:
        Dictionary mapping element -> propensity score.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    fe_freq = compute_frequencies(experimental_elements)
    fc_freq = compute_frequencies(comparison_elements)
    all_elements = sorted(set(fe_freq.keys()) | set(fc_freq.keys()))
    total = len(all_elements)

    n_exp = len(experimental_elements)
    n_comp = len(comparison_elements)
    n_total = n_exp + n_comp

    # --- GPU path ---
    if use_gpu and _gpu_available() and total > 0:
        print(f"    GPU acceleration enabled ({total} elements, "
              f"{n_permutations} permutations, {n_total} combined size)",
              flush=True)

        # Encode combined list as integer IDs
        elem_to_id = {e: i for i, e in enumerate(all_elements)}
        combined = experimental_elements + comparison_elements

        # Build binary masks: (n_elements, n_total)
        # mask[e, i] = 1 if combined[i] == all_elements[e]
        combined_ids = np.array(
            [elem_to_id.get(c, -1) for c in combined], dtype=np.int32
        )
        element_masks = np.zeros((total, n_total), dtype=np.bool_)
        for e_idx in range(total):
            element_masks[e_idx] = (combined_ids == e_idx)

        # Observed diffs
        observed_diffs = np.array([
            abs(fe_freq.get(e, 0.0) - fc_freq.get(e, 0.0))
            for e in all_elements
        ], dtype=np.float32)

        # Extract seed from rng for reproducibility
        seed = int(rng.integers(0, 2**31))

        pvals = _batch_permutation_test_gpu(
            combined_ids, n_exp, n_comp,
            element_masks, observed_diffs,
            n_permutations=n_permutations,
            seed=seed,
        )

        table = {}
        for i, elem in enumerate(all_elements):
            fe = fe_freq.get(elem, 0.0)
            fc = fc_freq.get(elem, 0.0)
            table[elem] = compute_propensity(fe, fc, float(pvals[i]),
                                             n_exp=n_exp, n_comp=n_comp)

        print(f"    GPU: done. Computed {total} propensity scores.", flush=True)
        return table

    # --- CPU path ---
    if total > 0:
        print(f"    CPU mode ({total} elements, {n_permutations} permutations)",
              flush=True)

    pvals = _batch_permutation_test_cpu(
        experimental_elements, comparison_elements,
        all_elements, n_permutations, rng,
    )

    table = {}
    for elem in all_elements:
        fe = fe_freq.get(elem, 0.0)
        fc = fc_freq.get(elem, 0.0)
        table[elem] = compute_propensity(fe, fc, pvals[elem],
                                         n_exp=n_exp, n_comp=n_comp)

    return table

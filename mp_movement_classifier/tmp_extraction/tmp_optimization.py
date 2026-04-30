"""Joint grid sweep over (num_t_points, num_mps) for the TMP model.

Trains one TMP model per (T, K) cell of the grid, records the Laplace
approximation to the log marginal likelihood (LAP) and the final VAF, writes
the full table to a text file, and produces two separate heatmap figures
(LAP and VAF), following the layout of Leh et al. (2023, Fig. 2).
"""

import os
from typing import List, Optional, Tuple

import numpy as np

from TMP_model import MP_model


def _train_one(
        processed_data: List[np.ndarray],
        num_mps: int,
        num_t_points: int,
        adam_steps: int,
        bfgs_steps: int,
) -> Tuple[float, float]:
    """Train a single TMP config; return (lap, vaf)."""
    num_segments = len(processed_data)
    num_signals = processed_data[0].shape[0]
    model = MP_model(
        num_t_points=num_t_points,
        num_MPs=num_mps,
        num_segments=num_segments,
        num_signals=num_signals,
        init_data=processed_data,
    )
    model.learn(processed_data, adam_steps=adam_steps, bfgs_steps=bfgs_steps)
    vaf = float(model.VAF_curve[-1])
    lap = float(model.Laplace_approx(processed_data))
    return lap, vaf


def _save_text_report(
        t_values: List[int],
        mps_values: List[int],
        lap_grid: np.ndarray,
        vaf_grid: np.ndarray,
        best_t: int,
        best_k: int,
        save_dir: str,
) -> str:
    path = os.path.join(save_dir, 'tmp_grid_optimization.txt')
    with open(path, 'w') as f:
        f.write("TMP joint grid sweep over (num_t_points, num_mps)\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"num_t_points values: {t_values}\n")
        f.write(f"num_mps values:      {mps_values}\n")
        f.write(f"Best (argmax LAP):   num_t_points={best_t}, num_mps={best_k}\n\n")

        f.write("LAP score grid (rows=num_t_points, cols=num_mps)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'T | K':<10}" + "".join(f"{k:>12}" for k in mps_values) + "\n")
        for i, t in enumerate(t_values):
            f.write(f"{t:<10}" + "".join(f"{lap_grid[i, j]:>12.2f}" for j in range(len(mps_values))) + "\n")

        f.write("\nVAF grid (rows=num_t_points, cols=num_mps)\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'T | K':<10}" + "".join(f"{k:>12}" for k in mps_values) + "\n")
        for i, t in enumerate(t_values):
            f.write(f"{t:<10}" + "".join(f"{vaf_grid[i, j]:>12.4f}" for j in range(len(mps_values))) + "\n")
    print(f"Results saved to: {path}")
    return path


def render_heatmap(
        grid: np.ndarray,
        t_values: List[int],
        mps_values: List[int],
        best_t: int,
        best_k: int,
        title: str,
        save_dir: str,
        file_stem: str,
        cmap: str = 'viridis',
) -> None:
    """Compact Leh-style heatmap with square cells.

    Single source of styling shared between training-time figures (called
    from ``optimize_tmp_grid``)
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    im = ax.imshow(grid, aspect='equal', origin='upper', cmap=cmap)

    ax.set_xticks(range(len(mps_values)))
    ax.set_xticklabels(mps_values, fontsize=11)
    ax.set_yticks(range(len(t_values)))
    ax.set_yticklabels(t_values, fontsize=11)
    ax.set_xlabel('num_mps (K)', fontsize=11)
    ax.set_ylabel('num_t_points (T)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Argmax-LAP cell, per Leh et al.
    j = mps_values.index(best_k)
    i = t_values.index(best_t)
    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                           fill=False, edgecolor='red', linewidth=2))

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.locator = MaxNLocator(nbins=4)
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    png_path = os.path.join(save_dir, f"{file_stem}.png")
    svg_path = os.path.join(save_dir, f"{file_stem}.svg")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure saved to: {png_path}")


# Back-compat alias
_save_heatmap = render_heatmap


def optimize_tmp_grid(
        processed_data: List[np.ndarray],
        t_values: List[int],
        mps_values: List[int],
        adam_steps: int = 100,
        bfgs_steps: int = 30,
        save_dir: Optional[str] = None,
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """
    Joint grid sweep over (num_t_points, num_mps).

    Returns
    -------
    best_t : int
    best_k : int
    lap_grid : np.ndarray, shape (len(t_values), len(mps_values))
    vaf_grid : np.ndarray, same shape as lap_grid
    """
    nT, nK = len(t_values), len(mps_values)
    lap_grid = np.full((nT, nK), -np.inf)
    vaf_grid = np.zeros((nT, nK))

    print("\n" + "=" * 70)
    print(f"TMP joint grid sweep: {nT} x {nK} = {nT * nK} fits")
    print(f"  num_t_points: {t_values}")
    print(f"  num_mps:      {mps_values}")
    print("=" * 70 + "\n")

    total = nT * nK
    step = 0
    for i, T in enumerate(t_values):
        for j, K in enumerate(mps_values):
            step += 1
            print(f"[{step}/{total}] T={T}, K={K}")
            try:
                lap, vaf = _train_one(processed_data, K, T, adam_steps, bfgs_steps)
            except Exception as e:
                print(f"  failed: {e}")
                lap, vaf = -np.inf, 0.0
            lap_grid[i, j] = lap
            vaf_grid[i, j] = vaf
            print(f"  LAP={lap:.2f}  VAF={vaf:.4f}")

            # Incrementally persist text report so partial runs are recoverable
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                i_best, j_best = np.unravel_index(np.argmax(lap_grid), lap_grid.shape)
                _save_text_report(
                    t_values, mps_values, lap_grid, vaf_grid,
                    best_t=t_values[i_best], best_k=mps_values[j_best],
                    save_dir=save_dir,
                )

    i_best, j_best = np.unravel_index(np.argmax(lap_grid), lap_grid.shape)
    best_t = t_values[i_best]
    best_k = mps_values[j_best]

    print(f"\nBest: num_t_points={best_t}, num_mps={best_k}")
    print(f"  LAP = {lap_grid[i_best, j_best]:.2f}")
    print(f"  VAF = {vaf_grid[i_best, j_best]:.4f}\n")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        _save_text_report(t_values, mps_values, lap_grid, vaf_grid, best_t, best_k, save_dir)
        render_heatmap(
            grid=lap_grid, t_values=t_values, mps_values=mps_values,
            best_t=best_t, best_k=best_k,
            title='LAP', cmap='viridis',
            save_dir=save_dir, file_stem='tmp_grid_lap',
        )
        render_heatmap(
            grid=vaf_grid, t_values=t_values, mps_values=mps_values,
            best_t=best_t, best_k=best_k,
            title='VAF', cmap='viridis',
            save_dir=save_dir, file_stem='tmp_grid_vaf',
        )

    return best_t, best_k, lap_grid, vaf_grid

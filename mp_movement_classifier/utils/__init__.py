"""Utils package."""

# Re-export everything from utils.py
from .utils import (
    config,
    load_model_with_full_state,
    # plot_learn_curve,
    # plot_mp,
    # plot_reconstructions,
    process_bvh_data,
    read_bvh_files,
    save_model_with_full_state,
    # set_figures_directory,
)

__all__ = [
    "config",
    "load_model_with_full_state",
    # "plot_learn_curve",
    # "plot_mp",
    # "plot_reconstructions",
    "process_bvh_data",
    "read_bvh_files",
    "save_model_with_full_state",
    "set_figures_directory",
]

# """Utils package for MP classifier."""
#
# # If you split utils.py into multiple files, import from them
# from .io import read_bvh_files, process_bvh_data
# from .model import load_model_with_full_state, save_model_with_full_state
# from .visualization import plot_learn_curve, plot_mp, plot_reconstructions, set_figures_directory
# from .config import config
#
# # Make all available at package level
# __all__ = [
#     "config",
#     "load_model_with_full_state",
#     "plot_learn_curve",
#     "plot_mp",
#     "plot_reconstructions",
#     "process_bvh_data",
#     "read_bvh_files",
#     "save_model_with_full_state",
#     "set_figures_directory",
# ]
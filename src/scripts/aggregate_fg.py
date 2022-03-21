
from resonant_lsm import utils

FILES = (
    "May16/cell_FG/cell_ref_FG_results/results.xlsx",
    "June14/cell_FG/cell_ref_FG_results/results.xlsx",
    "June21/cell_FG/cell_ref_FG_results/results.xlsx",
    "June28/cell_FG/cell_ref_FG_results/results.xlsx",
    "July04/cell_FG/cell_ref_FG_results/results.xlsx",
    "July16/cell_FG/cell_ref_FG_results/results.xlsx")

TIME_MAP = {
    "cell_2min_FG_results": 2,
    "cell_4min_FG_results": 4,
    "cell_6min_FG_results": 6,
    "cell_10min_FG_results": 10,
    "cell_16min_FG_results": 16,
    "cell_26min_FG_results": 26,
    "cell_41min_FG_results": 41,
    "cell_76min_FG_results": 76
}

utils.aggregate_strain_results(FILES, TIME_MAP, output_file="aggregated_strain_results_fg")
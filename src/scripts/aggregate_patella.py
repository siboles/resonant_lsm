from resonant_lsm import utils

FILES = (
    "../../dat/May16/cell_patella/cell_ref_patella_results/results.xlsx",
    "../../dat/June14/cell_patella/cell_ref_patella_results/results.xlsx",
    "../../dat/June21/cell_patella/cell_ref_patella_results/results.xlsx",
    "../../dat/June28/cell_patella/cell_ref_patella_results/results.xlsx",
    "../../dat/July04/cell_patella/cell_ref_patella_results/results.xlsx",
    "../../dat/July16/cell_patella/cell_ref_patella_results/results.xlsx")

TIME_MAP = {
    "cell_2min_patella_results": 2,
    "cell_4min_patella_results": 4,
    "cell_6min_patella_results": 6,
    "cell_10min_patella_results": 10,
    "cell_16min_patella_results": 16,
    "cell_26min_patella_results": 26,
    "cell_41min_patella_results": 41,
    "cell_76min_patella_results": 76
}

utils.aggregate_strain_results(FILES, TIME_MAP, output_file="aggregated_strain_results_patella")

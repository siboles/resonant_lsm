import pathlib
import shutil
import yaml
import sys


def main(yamlfile):
    with open(yamlfile) as f:
        cells_to_remove = yaml.load(f, yaml.SafeLoader)

    for specimen in cells_to_remove.keys():
        for region, cell_ids in cells_to_remove[specimen].items():
            print(specimen, region)
            p = pathlib.Path(specimen, region)
            directories = list(p.glob("**/*_results"))
            for directory in directories:
                bad_cell_directory = directory.joinpath("bad_cells")
                bad_cell_directory.mkdir(exist_ok=True)
                for cell_id in cell_ids:
                    bad_cells = list(directory.glob(f"*{cell_id:03d}.vtp"))
                    if bad_cells:
                        print(f"Moving {bad_cells[0]} to {bad_cell_directory}")
                        shutil.move(bad_cells[0], bad_cell_directory.joinpath(bad_cells[0].name))

if __name__ == "__main__":
    main(sys.argv[-1])

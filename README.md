# Description

This Python module is aimed at segmentation of cells from 3-D microscopy images 
acquired with resonance scanning. These images have generally lower quality than 
those acquired with traditional servo motor driven rasterization. 

# Installation

The easiest installation method is via conda, which requires the installation of either Anaconda3 or miniconda3

```
conda create -n resonant_lsm resonant_lsm
```

# Usage

From the command line:

```
python -m resonant_lsm.segmenter PATH_TO_CONFIG_FILE
```


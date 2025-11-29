# IntroDS Project

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate introds_project
```

Using mamba (recommended for speed):

```bash
# Install mamba into the base environment if you don't have it
conda install -n base -c conda-forge mamba

# prefer packages from higher-priority channels (recommended)
conda config --set channel_priority strict

# Create the environment with mamba and activate it
mamba env create -f environment.yml
conda activate introds_project
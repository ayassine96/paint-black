# Paint-it-Gray: Modeling, Partitioning, and Analysis of User Transaction Networks in Bitcoin

This repository contains the code and data related to the Master Thesis titled **"Paint-it-Gray: Modeling, Partitioning, and Analysis of User Transaction Networks in the Bitcoin Blockchain"** by Ali Yassine.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Scripts Description](#scripts-description)
- [Data Sources](#data-sources)
- [Results and Analysis](#results-and-analysis)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Introduction
This project introduces the "grayscale diffusion framework" for modeling and understanding the propagation of dark assets in the Bitcoin network. By utilizing address clustering, haircut tainting, and community partitioning, this framework offers a unique analysis perspective on dark asset proliferation across the Bitcoin blockchain.

## Project Structure
The repository is structured as follows:
- `.gitignore`: Specifies files and directories to be ignored by git.
- `Gini_entropy_builder.py`: Script for building Gini entropy measures.
- `Gini_entropy_random_builder.py`: Script for building randomized Gini entropy measures.
- `assortativity_builder.py`: Script for building assortativity measures.
- `assortativity_random_builder.py`: Script for building randomized assortativity measures.
- `community_builder.py`: Script for building community structures.
- `gs_diffusion_daily_weekly_final.py`: Main script for grayscale diffusion on a daily and weekly basis.
- `gs_ground_truth.py`: Script for establishing ground truth data.
- `gs_plotting_weekly_final.py`: Script for plotting weekly grayscale diffusion results.
- `jsonResults_v3`: Directory containing JSON results.
- `nb_graph_scripts copy.ipynb`: Jupyter notebook with graph scripts (copy).
- `nb_graph_scripts.ipynb`: Jupyter notebook with graph scripts.
- `nb_scratchbook.ipynb`: Jupyter notebook for scratch work.
- `nb_uniform_black_sanity-check.ipynb`: Jupyter notebook for uniform black sanity check.
- `nb_zarr_test.ipynb`: Jupyter notebook for Zarr testing.
- `network_builder.py`: Script for building network structures.
- `obsolete`: Directory containing obsolete scripts and files.
- `readme.md`: This README file.
- `shell_scripts`: Directory containing shell scripts.
- `util.py`: Utility functions.
- `walletexplorer_DNM`: Directory containing wallet explorer darknet market data.

## Setup and Installation
To set up and run the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/paint-black-master.git
   cd paint-black-master
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To run the various scripts, you can use the following commands. Make sure you have activated your virtual environment.

For example, to run the grayscale diffusion script:
```bash
python gs_diffusion_daily_weekly_final.py
```

## Scripts Description
- **Gini_entropy_builder.py**: Builds Gini entropy measures to analyze the distribution of assets.
- **Gini_entropy_random_builder.py**: Creates randomized Gini entropy measures for comparison.
- **assortativity_builder.py**: Constructs assortativity measures to study the mixing patterns of nodes.
- **assortativity_random_builder.py**: Generates randomized assortativity measures.
- **community_builder.py**: Builds community structures within the transaction network.
- **gs_diffusion_daily_weekly_final.py**: Implements the grayscale diffusion model to track dark asset propagation.
- **gs_ground_truth.py**: Establishes ground truth data for validation.
- **gs_plotting_weekly_final.py**: Plots the results of the grayscale diffusion on a weekly basis.

## Data Sources
The data used in this project includes both on-chain and off-chain data:
- On-chain data from the Bitcoin blockchain.
- Off-chain data from darknet markets and wallet explorer.

## Results and Analysis
The results of this project provide insights into the propagation of dark assets in the Bitcoin network, including:
- Analysis of dark asset diffusion.
- Community structures and their roles in dark asset propagation.
- Assortative mixing patterns based on darkness levels.

## Contributing
Contributions are welcome! Please read the [contributing guidelines](CONTRIBUTING.md) before submitting a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
This project was developed as part of the Master Thesis by Ali Yassine under the supervision of Prof. Dr. Claudio J. Tessone and Dr. Vallarano Nicolò at the University of Zurich.

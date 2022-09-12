# Code for "Iterative training of robust k-space interpolation networks for improved image reconstruction with limited scan specific training samples"
[Paper iterativeRAKI (iRAKI)](https://arxiv.org/abs/2201.03560). Please feel free to ask questions via email (peter.dawood@physik.uni-wuerzburg.de).

## Installation

### Conda

For normal training and evaluation we recommend to install the package from source via a `conda` virtual environment.
This facilitates reproducibility and maximum isolation from unwanted interference with other packages.
To do so, first install a Python distribution like [Anaconda](https://www.anaconda.com/products/distribution) or
[Miniconda](https://docs.conda.io/en/latest/miniconda.html).
Then clone this repository and automatically create a new, isolated environment `iteraki` and install the dependencies
via a single CLI command.
After activating the envioronment, you are ready to start with training and evaluation.
```
git clone https://github.com/pdawood/iterativeRaki.git
cd iterativeRaki
conda env create -f environment.yaml
conda activate iteraki
```

## Sample Datasets


### In-line calibration
The jupyter-notebook *inLine.ipynb* demonstrates the reconstruction of a uniform 4-fold undersampled 2D image using only 18 ACS lines as training data: 

![Alt text](/data/inlineR4.png)

Optionally, a phase-constraint via virtual-conjugate-coils can be included to improve reconstruction quality:
![Alt text](/data/inlineR4VCC.png)
### Pre-scan calibration
The jupyter-notebook 'preScan.ipynb' demonstrates the reconstruction of a uniform 4-fold undersampled 2D image using a pre-scan as training data. The pre-scan has matrix size 64x64 and different contrast information than the target image:
![Alt text](/data/prescanRef.png)
![Alt text](/data/prescanR4.png)


## License
We release the code under the [MIT License](https://en.wikipedia.org/wiki/MIT_License)


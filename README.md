![Python](https://img.shields.io/badge/python-v3.6.5-green.svg)
![JupyterNotebook](https://img.shields.io/badge/jupyter-v4.6.1-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-v1.4.0-blue.svg)
# PROCAT: Product Catalogue Dataset for Structure Prediction

### Introduction
This repository contains the code and dependency specification required for running repeated experiments on the titular dataset, which is publicly available under the CC BY-NC-SA license [here](https://doi.org/10.6084/m9.figshare.14709507).

PROCAT is a dataset of over 10,000 product catalogues consisting of more than 1.5 million individual product offers. This dataset lends itself to machine learning research in the area of set-to-sequence structure prediction, clustering and permutation learning.

It contains the **text** features of offers grouped into sections such as these:
<p float="left">
  <img src="./img/sample_catalog_section_1.png" width="250" />
  <img src="./img/sample_catalog_section_2.png" width="250" /> 
  <img src="./img/sample_catalog_section_3.png" width="250" />
</p>

### Content

The repository consists primarily of two jupyter notebooks for repeated experiments.

The first one, titled **procat_experiments.ipynb** contains training and evaluation code for models trained on the main PROCAT dataset. In order to run it, you first need to download it via this [dataset link](https://doi.org/10.6084/m9.figshare.14709507), unzip it and place it in the proper source folder.

The second one, titled **synthetic_experiments.ipynb** contains training and evaluation code for models trained on a synthetically generated set of simplified catalogue structures.

Each notebook starts with configuration information, which you can adjust.

### License

This content is made available under the CC BY-NC-SA license.

![Python](https://img.shields.io/badge/python-v3.6.5-green.svg)
![JupyterNotebook](https://img.shields.io/badge/jupyter-v4.6.1-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-v1.4.0-blue.svg)
# PROCAT: Product Catalogue Dataset for Structure Prediction

### Introduction
This repository contains the code and dependency specification required for running repeated experiments on the titular dataset, which is publicly available under the CC BY-NC-SA license at this [PROCAT dataset link](https://doi.org/10.6084/m9.figshare.14709507).

PROCAT is a dataset of over 10,000 product catalogues consisting of more than 1.5 million individual product offers. This dataset lends itself to machine learning research in the area of set-to-sequence structure prediction, clustering and permutation learning.

It contains the **text** features of offers grouped into sections such as these:
<p float="left">
  <img src="./img/sample_catalog_section_1.png" width="250" />
  <img src="./img/sample_catalog_section_2.png" width="250" /> 
  <img src="./img/sample_catalog_section_3.png" width="250" />
</p>

For details regarding the format, main features and the structure of the dataset files, see the [dataset structure](#dataset-structure) section.

### Content

The repository consists primarily of two jupyter notebooks for repeated experiments.

The first one, titled [procat experiments notebook](procat_experiments.ipynb) contains training and evaluation code for models trained on the main PROCAT dataset. In order to run it, you first need to download it via this [dataset link](https://doi.org/10.6084/m9.figshare.14709507), unzip it and place it in the proper source folder.

The second one, titled [synthetic experiments notebook](synthetic_experiments.ipynb) contains training and evaluation code for models trained on a synthetically generated set of simplified catalogue structures.

Each notebook starts with configuration information, which you can adjust.

Additionally, the best performing model is made available with the DOI `10.5281/zenodo.4896303` at this [model hosting address](https://zenodo.org/record/4896303#.YLnxgZMzbOQ). The model requires using the appropriate version of PyTorch and importing the proper model definition, as shown in the provided jupyter notebooks.

### Usage

You must have the proper version of python installed (3.6.5). Then, create a virtual environment:

`python3 -m venv /path/to/new/virtual/environment`

Once the virtual environment has been installed, activate it (this may differ depending on your operating system):

`source <ven_pathv>/bin/activate`

Then, from within the environemnt, install all requirements from the provided file via:

`python3 -m pip install -r requirements.txt`

Finally, activate jupyter via the followind terminal command:

`jupyter notebook` 

or:

`jupyter lab`

depending on your preferred user interface.

This should start a jupyter process, allowing you to visit `localhost:8888` and run the provided notebooks.

In order to run the main **procat_experiments.ipynb** notebook, you will need to download the dataset and unzip it, resulting in a `PROCAT` folder with the necessary `.csv`, `.pickle`, `.pb` and `.npy` files being in the same directory as the notebooks.

This should result in the following output of the `tree` command (or equivalent on your OS), when run from the directory of this repository:

```asciidoc
├── PROCAT
│   ├── X_test.npy
│   ├── X_test.pb
│   ├── X_train.npy
│   ├── X_train.pb
│   ├── Y_test.npy
│   ├── Y_test.pb
│   ├── Y_train.npy
│   ├── Y_train.pb
│   ├── catalog_features.csv
│   ├── catalog_test_set_features.csv
│   ├── catalog_to_sections.pickle
│   ├── catalog_train_set_features.csv
│   ├── dictionary.pickle
│   ├── offer_features.csv
│   ├── offer_to_priority.pickle
│   ├── offer_to_vector.pickle
│   ├── section_features.csv
│   ├── section_id_to_offer_priorities.pickle
│   ├── section_id_to_offer_vectors.pickle
│   ├── section_to_number.pickle
│   └── section_to_offers.pickle
├── PROCAT_mini
├── README.md
├── data.py
├── img
├── models.py
├── procat_experiments.ipynb
├── procat_models.py
├── procat_utils.py
├── requirements.txt
├── synthetic_experiments.ipynb
├── synthetic_functional.py
└── utils.py

```

If you wish to store & view repeated experimental results, you will also need to set up a Mongo database (instructions [here](https://docs.mongodb.com/manual/installation/#std-label-tutorial-installation)) and have a local omniboard instance running (more information [here](https://github.com/vivekratnavel/omniboard)). Alternatively, you'll need to control which cells are executed and skip the ones relating to tracking the experiment results via the **sacred** library. You can get the same metrics by running the cells marked as raw within the notebook.

### Dataset Structure

If you prefer not to use the provided jupyter notebooks, the easiest way to get an overview of the files that form the PROCAT dataset is to start by viewing the `PROCAT/offer_features.csv` file, or the `PROCAT_mini/offer_features.csv` which contains a small subset of the data in the exact same format, for quick tests.

Every product offer instance consists of:

* a *catalog_id* identifying the catalog that the offer belonged to.
* a *section* marker in the form of an integer, identifying the ordered section in which the offer was placed in the original catalog.
* an *offer_id* uniquely identifying a single product offer.
* a *priority* rating, on a 1-3 integer scale, marking the relative in-section visual prominence of the product offer's image (primarily defined by relative size).
* a *heading* text header of the offer.
* a *description* text description of the offer.
* a *text* field, with concatenated heading and description.
* a *tokenized_text*, which consists of word tokens obtained via the appropriate NLTK tokenizer (for details see the paper).
* a *token_length* field, measuring the number of word tokens.
* an *offer_as_vector*, containing an array of dictionary-based integers representing words from the vocabulary.

The aforementioned vocabulary mapping integers to word tokens is available as the `PROCAT/dictionary.pickle` file, along with a number of helper python dictionaries:

1. `PROCAT/offer_to_priority.pickle` mapping offer ids to their in-section *priority* class.
2. `PROCAT/offer_to_vector.pickle` mapping offer ids to their vector representations.
3. `PROCAT/section_id_to_offer_priorities.pickle` mapping section ids to an ordered list of offer priorities.
4. `PROCAT/section_id_to_offer_vectors.pickle` mapping section ids to the vector representations of their offers, in order.
5. `PROCAT/section_to_offers.pickle` mapping section ids to a list of offers that comprised them.

These mappings are also available from within the `PROCAT/section_features.csv` file, which maps section ids to their corresponding catalogue ids and offer features (as a list of offer ids, offer vectors and priorities).

Finally, the main `PROCAT/catalog_features.csv` file contains our main X for experiments. Each row is a single catalogue, consisting of:

* the catalogue id.
* a list of its section ids, in order.
* a list of its offer ids, in order, with section break markers.
* a 2-dimensional array of its offers as vectors, with section break tokens.
* a list of its offer priorities, in order.
* the total number of offers per catalogue.
* a single `x` matrix, ready as input to the provided set-to-sequence models, representing one random permutation of the offers vectors.
* a single `y` array, representing the correct permutation restoring the original order of offers in the catalogue.

### License

This content is made available under the CC BY-NC-SA license.

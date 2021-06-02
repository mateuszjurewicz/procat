"""
Script for other useful functions for pre- and post processing the real catalog
data, obtained as a csv by the publication_2_data_real_catalogs_extraction.py.
"""

import numpy as np
import pandas as pd
import pickle
import torch
import ast
import random


def get_catalogs_with_full_info(a_dataframe, print_every=5000):
    """
    Take a dataframe as given by the publication_2_data_real_catalogs_extraction.py
    and return a set containing IDs of catalogs with no missing text (either
    heading or description are present).
    """
    catalog_ids_with_missing_data = set()
    catalog_ids_all = set()
    counter = 0
    total_rows = len(a_dataframe.index)

    for r in a_dataframe.iterrows():

        # ignore index, get the data
        row_data = r[1]

        catalog_id = row_data['catalog_id']
        heading = row_data['heading']
        description = row_data['description']

        catalog_ids_all.add(catalog_id)

        if pd.isnull(heading) and pd.isnull(description):
            catalog_ids_with_missing_data.add(catalog_id)
        else:
            pass

        # report progress
        counter += 1
        if counter % print_every == 0:
            print('Processing row {} / {}'.format(counter, total_rows))

    print('Total catalogs: ', len(catalog_ids_all))
    print('Missing some data catalogs: ', len(catalog_ids_with_missing_data))
    print('Usable catalogs with full data: ',
          len(catalog_ids_all) - len(catalog_ids_with_missing_data))

    # get ids for catalogs with no missing data
    catalog_ids_full_data = catalog_ids_all.difference(
        catalog_ids_with_missing_data)

    print(len(catalog_ids_full_data))

    return catalog_ids_full_data


def show_correct_catalog(catalog_id, catalogs_dataframe, offers_dataframe):
    """
    Print the chosen catalog in human-readable way.
    """

    correct_catalog_as_offer_tokens = ast.literal_eval(catalogs_dataframe.loc[catalogs_dataframe['catalog_id'] == catalog_id]['offer_ids_with_pb'][0])
    print_predicted_catalog(correct_catalog_as_offer_tokens, offers_dataframe)


def show_predicted_catalog(catalog_id, a_model, catalogs_dataframe,
                           offers_dataframe):
    """
    Take a catalog id and show the order of offers predicted by the model.
    """
    # put model in eval mode
    a_model.eval()

    # find catalog offers
    chosen_offer_ids = ast.literal_eval(
        catalogs_dataframe.loc[catalogs_dataframe['catalog_id'] == catalog_id][
            'offer_ids_with_pb'][0])
    chosen_offer_vectors = ast.literal_eval(
        catalogs_dataframe.loc[catalogs_dataframe['catalog_id'] == catalog_id][
            'offer_vectors_with_pb'][0])

    # define a random permutation
    chosen_permutation = [i for i in range(len(chosen_offer_ids))]
    random.shuffle(chosen_permutation)

    # shuffle offer vectors into an x
    permuted_x = [chosen_offer_vectors[i] for i in chosen_permutation]

    # shuffle offer ids
    permuted_offer_ids = [chosen_offer_ids[i] for i in chosen_permutation]

    # to numpy
    permuted_x = np.array(list(permuted_x), np.int32)

    # to torch
    permuted_x = torch.from_numpy(np.asarray(permuted_x))

    # predict
    _, a_prediction = a_model(permuted_x.unsqueeze(0))

    # grab actual correct y, for error-checking
    correct_y = ['to_be_replaced'] * len(chosen_permutation)
    for i, e in enumerate(chosen_permutation):
        correct_y[e] = i

    # get prediction as offer tokens in order
    predicted_catalog = get_prediction_as_offers(a_prediction[0],
                                                 permuted_offer_ids)

    # display nicely
    print_predicted_catalog(predicted_catalog, offers_dataframe)


# use the prediction to reconstruct predicted order of offer ids
def get_prediction_as_offers(prediction, permuted_offers):
    """
    Take permuted offer ids and a pointer-style prediction,
    restore predicted order.
    """
    r = ['tbr'] * len(permuted_offers)
    for i, e in enumerate(prediction):
        r[i] = permuted_offers[e]
    return r


def print_predicted_catalog(a_catalog, offer_features_df):
    """
    Nicely display the prediction, based on text features of offers.
    """
    # display knobs
    h_idx = 40
    d_idx = 40

    # iterate
    for offer in a_catalog:
        # handle special tokens
        if offer == '?NOT_REAL_OFFER?':
            print('\t', offer)
        elif offer == '?PAGE_BREAK?':
            print('------------------------')
        else:
            # get header and description
            header = \
            offer_features_df.loc[offer_features_df['offer_id'] == offer][
                'text'].values[0]
            description = \
            offer_features_df.loc[offer_features_df['offer_id'] == offer][
                'description'].values[0]

            # clear NaN
            if header is np.NaN:
                header = ''
            if description is np.NaN:
                description = ''
            print('\t{} h: {}, d: {}'.format(offer, header[:h_idx],
                                             description[:d_idx]))


if __name__ == '__main__':

    # path to output
    path_to_output = 'data/real_catalogs/catalog_ids_no_missing_text_data.pkl'

    # path to input csv, with catalogs, sections, priority and text features
    path_to_input = 'data/real_catalogs/catalog_page_to_offer_priority_' \
                    'and_features_60000_catalogs.csv'

    # open the csv with catalogs with missing data
    df = pd.read_csv(path_to_input, delimiter=';')

    # get catalogs with no text missing
    catalog_ids_no_missing = get_catalogs_with_full_info(df)

    # persist the valid catalog ids for later
    with open(path_to_output, 'wb') as f:
        pickle.dump(catalog_ids_no_missing, f)



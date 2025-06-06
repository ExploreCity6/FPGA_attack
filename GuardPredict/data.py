# -*- coding: utf-8 -*-
import logging
import os
import pickle
# import ujson as json

import numpy as np
from scipy import sparse


def load_features(dataset, folder='features/'):
    logging.info('Loading ' + dataset + '_gw features...')
    filepath = os.path.join(folder, dataset + '_gw_features.p')
    data_gw = load_csr_list(filepath)
    labels_gw = [0] * data_gw.shape[0]

    logging.info('Loading ' + dataset + '_mw features...')
    filepath = os.path.join(folder, dataset + '_mw_features.p')
    data_mw = load_csr_list(filepath)
    labels_mw = [1] * data_mw.shape[0]

    X = sparse.vstack([data_gw, data_mw], format='csr')
    y = np.array(labels_gw + labels_mw)

    return X, y


def load_csr_list(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, list):
        data = sparse.vstack(data, format='csr')

    return data


def load_cached_data(data_path):
    logging.info('Loading data from {}...'.format(data_path))
    with open(data_path, 'rb') as f:
        model = pickle.load(f)
    logging.debug('Done.')
    return model


def cache_data(model, data_path):
    model_folder_path = os.path.dirname(data_path)

    if not os.path.exists(model_folder_path):
        os.makedirs(model_folder_path)

    logging.info('Saving data to {}...'.format(data_path))
    with open(data_path, 'wb') as f:
        pickle.dump(model, f)
    logging.debug('Done.')


# def save_results(results, args, output='txt'):
#     """Helper function to save results with distinct filename based on args."""
#     test_file, train_file = args.test.replace('/', ''), args.train.replace('/', '')
#
#     filename = 'report-{}-{}-fold-{}-{}-{}'.format(
#         train_file, args.folds, args.thresholds, args.criteria, test_file)
#     if args.criteria == 'quartiles':
#         filename += '-{}'.format(args.q_consider)
#     if args.criteria == 'constrained-search':
#         filename += '-{}max-{}con-{}s'.format(
#             args.cs_max, args.cs_min, args.rs_samples)
#     if args.criteria == 'random-search':
#         filename += '-{}max-{}mwrej-{}s'.format(
#             args.rs_max, args.rs_reject_limit, args.rs_samples)
#     filename += '.' + output
#
#     folder = './reports'
#
#     if not os.path.exists(folder):
#         os.makedirs(folder)
#
#     results_path = os.path.join(folder, filename)
#     logging.info('Saving results to {}...'.format(results_path))
#
#     if output == 'txt':
#         with open(results_path, 'wt') as f:
#             f.write(results)
#     elif output == 'json':
#         with open(results_path, 'wt') as f:
#             json.dump(results, f)
#     elif output == 'p':
#         with open(results_path, 'wb') as f:
#             pickle.dump(results, f)
#     else:
#         msg = 'Unknown file format could not save.'
#         logging.error(msg)
#     logging.debug('Done.')

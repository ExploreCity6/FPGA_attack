# -*- coding: utf-8 -*-
import logging
import statistics

import numpy as np
from sklearn import metrics as metrics
from termcolor import cprint
from tqdm import tqdm
from itertools import repeat

import multiprocessing as mp
import os


def test_with_rejection(
        multi_thresholds, test_scores, groundtruth_labels, predicted_labels, full=True):
    keep_mask = apply_threshold(
        multi_thresholds=multi_thresholds,
        test_scores=test_scores,
        y_test=predicted_labels)

    results = get_performance_with_rejection(
        y_true=groundtruth_labels,
        y_pred=predicted_labels,
        keep_mask=keep_mask,
        full=full)

    return results, keep_mask, predicted_labels


def apply_threshold(multi_thresholds, test_scores, y_test):
    # Assert preconditions
    assert (set(multi_thresholds.keys()) in
            [{'cred'}, {'conf'}, {'cred', 'conf'}])

    for key in multi_thresholds.keys():
        assert key in test_scores.keys()
        # assert set(binary_thresholds[key].keys()) == {'mw', 'gw'}

    def get_class_threshold(criteria, k):
        return multi_thresholds[criteria][f'class{k}']
        # return (binary_thresholds[criteria]['mw'] if k == 1
        #         else binary_thresholds[criteria]['gw'])

    keep_mask = []
    for i, y_prediction in enumerate(y_test):

        cred_threshold, conf_threshold = 0, 0
        current_cred, current_conf = 0, 0

        if 'cred' in multi_thresholds:
            key = 'cred'
            current_cred = test_scores[key][i]
            cred_threshold = get_class_threshold(key, y_prediction)

        if 'conf' in multi_thresholds:
            key = 'conf'
            current_conf = test_scores[key][i]
            conf_threshold = get_class_threshold(key, y_prediction)

        keep_mask.append(
            (current_cred >= cred_threshold) and
            (current_conf >= conf_threshold))

    return np.array(keep_mask, dtype=bool)


def find_quartile_thresholds(
        scores, predicted_labels, groundtruth_labels, consider='correct'):
    scores_list = sort_by_predicted_label(
        scores, predicted_labels, groundtruth_labels, consider=consider)
    temp_dict = {}
    for i in range(len(np.unique(groundtruth_labels))):
        temp_dict[f'class{i}'] = np.percentile(scores_list[i], 5) if len(scores_list[i]) > 0 else 0
    thresholds = {
        # 'q1': {
        #     'mw': np.percentile(scores_mw, 25),
        #     'gw': np.percentile(scores_gw, 25)
        # },
        # 'q2': {
        #     'mw': np.percentile(scores_mw, 50),
        #     'gw': np.percentile(scores_gw, 50)
        # },
        # 'q3': {
        #     'mw': np.percentile(scores_mw, 75),
        #     'gw': np.percentile(scores_gw, 75)
        # },
        # 'mean': {
        #     'mw': np.mean(scores_mw),
        #     'gw': np.mean(scores_gw)
        # },
        # 'aus1': {
        #     'mw': np.percentile(scores_mw, 5) if len(scores_mw) > 0 else 0,
        #     'gw': np.percentile(scores_gw, 5) if len(scores_gw) > 0 else 0
        # }
        'aus1': temp_dict
        # 'aus2': {
        #     'mw': np.percentile(scores_mw, 5),
        #     'gw': np.percentile(scores_gw, 10)
        # },
        # 'aus3': {
        #     'mw': np.percentile(scores_mw, 1),
        #     'gw': np.percentile(scores_gw, 10)
        # }
    }
    # print(thresholds)
    return thresholds


def sort_by_predicted_label(
        scores, predicted_labels, groundtruth_labels, consider='correct'):

    def predicted(i, k):
        return predicted_labels[i] == k

    def correct(i, k):
        return predicted(i, k) and (groundtruth_labels[i] == k)

    def incorrect(i, k):
        return predicted(i, k) and (groundtruth_labels[i] == (k ^ 1))

    if consider == 'all':
        select = predicted
    elif consider == 'correct':
        select = correct
    elif consider == 'incorrect':
        select = incorrect
    else:
        raise ValueError('Unknown thresholding criteria!')

    scores_list = []
    uni_labels = np.unique(groundtruth_labels)
    for kkk in range(len(uni_labels)):
        scores_list.append(np.array([scores[i] for i in range(len(scores)) if select(i, kkk)]))
    return scores_list
    # scores_mw = [scores[i] for i in range(len(scores)) if select(i, 1)]
    # scores_gw = [scores[i] for i in range(len(scores)) if select(i, 0)]
    #
    # return np.array(scores_mw), np.array(scores_gw)


def find_random_search_thresholds_with_constraints(
        scores, predicted_labels, groundtruth_labels, maximise_vals,
        constraint_vals, max_samples=100, quiet=False, ncpu=-1, full=True):
    ncpu = mp.cpu_count() + ncpu if ncpu < 0 else ncpu

    if ncpu == 1:
        results, thresholds = find_random_search_thresholds_with_constraints_discrete(
            scores, predicted_labels, groundtruth_labels, maximise_vals,
            constraint_vals, max_samples, quiet, full)

        return thresholds

    samples = [max_samples // ncpu for _ in range(ncpu)]

    with mp.Pool(processes=ncpu) as pool:
        results = pool.starmap(find_random_search_thresholds_with_constraints_discrete,
                               zip(repeat(scores), repeat(predicted_labels), repeat(groundtruth_labels),
                                   repeat(maximise_vals), repeat(constraint_vals), samples, repeat(quiet),
                                   repeat(full)))
        results_list = [res[0] for res in results]
        thresholds_list = [res[1] for res in results]

    def resolve_keyvals(s):
        if isinstance(s, str):
            pairs = s.split(',')
            pairs = [x.split(':') for x in pairs]
            return {k: float(v) for k, v in pairs}
        return s

    print('[*] 多cpu数据汇总中:')
    maximise_vals = resolve_keyvals(maximise_vals)
    constraint_vals = resolve_keyvals(constraint_vals)

    best_maximised = {k: 0 for k in maximise_vals}
    best_constrained = {k: 0 for k in constraint_vals}
    best_thresholds, best_result = {}, {}

    # 汇总不同cpu数据池
    for result, thresholds in zip(results_list, thresholds_list):
        if any([result[k] < constraint_vals[k] for k in constraint_vals]):
            continue

        if any([result[k] < best_maximised[k] for k in maximise_vals]):
            continue
        # 悲观的小提升
        if all([result[k] == best_maximised[k] for k in maximise_vals]):
            if all([result[k] >= best_constrained[k] for k in constraint_vals]):
                best_maximised = {k: result[k] for k in maximise_vals}
                best_constrained = {k: result[k] for k in constraint_vals}
                best_thresholds = thresholds
                best_result = result

                if not quiet != quiet:
                    logging.info('New best: {}|{}\n    @ {} '.format(
                        format_opts(maximise_vals.keys(), result),
                        format_opts(constraint_vals.keys(), result),
                        best_thresholds))
                    report_results(d=best_result, full=full)
            continue

        # 乐观的大提升
        if any([result[k] > best_maximised[k] for k in maximise_vals]):
            best_maximised = {k: result[k] for k in maximise_vals}
            best_constrained = {k: result[k] for k in constraint_vals}
            best_thresholds = thresholds
            best_result = result

            if not quiet != quiet:
                logging.info('New best: {}|{} \n    @ {} '.format(
                    format_opts(maximise_vals.keys(), result),
                    format_opts(constraint_vals.keys(), result),
                    best_thresholds))
                report_results(d=best_result, full=full)

            continue

    if best_thresholds == {}:
        print('找不到合适的阈值,尝试修改con阈值限制或更改模型gamma、C参数\n'
              '或提高sample数目或修改random_threshold()阈值随机范围')
        best_thresholds = {'cred': {'mw': 1, 'gw': 1}}

    print(best_thresholds)
    return best_thresholds


def find_random_search_thresholds_with_constraints_discrete(
        scores, predicted_labels, groundtruth_labels, maximise_vals,
        constraint_vals, max_samples=100, quiet=False, full=True, stop_condition=3000):
    # as this method is called from multiprocessing, we want to make sure each
    # process has a different seed 
    seed = 0
    for l in os.urandom(10): seed += l
    np.random.seed(seed)

    def resolve_keyvals(s):
        if isinstance(s, str):
            pairs = s.split(',')
            pairs = [x.split(':') for x in pairs]
            return {k: float(v) for k, v in pairs}
        return s

    maximise_vals = resolve_keyvals(maximise_vals)
    constraint_vals = resolve_keyvals(constraint_vals)

    best_maximised = {k: 0 for k in maximise_vals}
    best_constrained = {k: 0 for k in constraint_vals}
    best_thresholds, best_result = {}, {}

    logging.info('Searching for threshold on calibration data...')

    stop_counter = 0

    for _ in tqdm(range(max_samples)):
        # Choose and package random thresholds
        thresholds = {}
        if 'cred' in scores:
            cred_thresholds = random_threshold(scores['cred'], predicted_labels)
            thresholds['cred'] = cred_thresholds
        if 'conf' in scores:
            conf_thresholds = random_threshold(scores['conf'], predicted_labels)
            thresholds['conf'] = conf_thresholds

        # Test with chosen thresholds
        result, h, g = test_with_rejection(
            thresholds, scores, groundtruth_labels, predicted_labels, full)

        # Check if any results exceed given constraints (e.g. too many rejects)
        if any([result[k] < constraint_vals[k] for k in constraint_vals]):
            if stop_counter > stop_condition:
                logging.info('Exceeded stop condition, terminating calibration search...')
                break

            stop_counter += 1
            continue

        if any([result[k] < best_maximised[k] for k in maximise_vals]):
            if stop_counter > stop_condition:
                logging.info('Exceeded stop condition, terminating calibration search...')
                break

            stop_counter += 1
            continue

        # 悲观的小提升
        if all([result[k] == best_maximised[k] for k in maximise_vals]):
            if all([result[k] >= best_constrained[k] for k in constraint_vals]):
                best_maximised = {k: result[k] for k in maximise_vals}
                best_constrained = {k: result[k] for k in constraint_vals}
                best_thresholds = thresholds
                best_result = result

                print('\n[*] New best: {}|{}\n    @ {} '.format(
                    format_opts(maximise_vals.keys(), result),
                    format_opts(constraint_vals.keys(), result),
                    best_thresholds))
                if not quiet:
                    report_results(d=best_result, full=full)

            stop_counter = 0
            continue

        # 乐观的大提升
        if any([result[k] > best_maximised[k] for k in maximise_vals]):
            best_maximised = {k: result[k] for k in maximise_vals}
            best_constrained = {k: result[k] for k in constraint_vals}
            best_thresholds = thresholds
            best_result = result

            print('\n[*] New best: {}|{}\n    @ {} '.format(
                format_opts(maximise_vals.keys(), result),
                format_opts(constraint_vals.keys(), result),
                best_thresholds))
            if not quiet:
                report_results(d=best_result, full=full)

            stop_counter = 0
            continue

    if not bool(best_result):
        best_result = result

    return (best_result, best_thresholds)


def random_threshold(scores, predicted_labels):
    scores_mw, scores_gw = sort_by_predicted_label(
        scores, predicted_labels, np.array([]), 'all')

    mw_threshold = np.random.uniform(min(scores_mw), max(scores_mw))
    gw_threshold = np.random.uniform(min(scores_gw), max(scores_gw))
    #######################
    # print(mw_threshold, gw_threshold)
    return {'mw': mw_threshold, 'gw': gw_threshold}


def get_performance_with_rejection(y_true, y_pred, keep_mask, full=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    d = {}

    total = len(y_pred)

    kept_total = sum(keep_mask)
    reject_total = total - kept_total

    d.update({'total': total,
              'kept_total': kept_total, 'reject_total': reject_total,
              'kept_total_perc': kept_total / total,
              'reject_total_perc': reject_total / total,
              })

    return d


def report_results(d, quiet=False, full=False):
    report_str = ''

    def print_and_extend(report_line):
        nonlocal report_str
        if not quiet:
            cprint(report_line, 'yellow')
        report_str += report_line + '\n'

    s = '% kept elements(kt):   \t{:6d}/{:6d} = {:.1f}%, \t% rejected elements(rt):   \t{:6d}/{:6d} = {:.1f}%'.format(
        d['kept_total'], d['total'], d['kept_total'] / d['total'] * 100,
        d['reject_total'], d['total'], d['reject_total'] / d['total'] * 100
    )
    print_and_extend(s)

    return report_str


def format_opts(metrics, results):
    return (' {}: {:.4f} |' * len(metrics)).format(
        *[item for sublist in
          zip(metrics, [results[k] for k in metrics]) for
          item in sublist])

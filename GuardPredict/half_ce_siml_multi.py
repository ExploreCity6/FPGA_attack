import csv
import logging
import os

import pandas as pd
import numpy as np
from termcolor import cprint
import datetime

import data as data
import thresholding_multi as thresholding
import utils as utils
import scores_multi as scores


def start_half_transcend(_train_probs, _train_y_true,
                         _cal_probs, _cal_y_true, _cal_y_pred,
                         _test_probs, _test_y_true, _test_y_pred,
                         _saved_data_folder='', tostore=False
                         ):
    start_time = datetime.datetime.now()

    write_time = str(start_time.today())
    write_time = write_time.replace(':', '_')
    print('开始计时：{}'.format(start_time))

    # utils.configure_logger()
    args = utils.parse_args()
    _saved_data_folder += '-{}'.format(write_time)

    if tostore:
        saved_pvals_name = 'p_val_cal_fold_dict.p'
        saved_pvals_name = os.path.join(_saved_data_folder, saved_pvals_name)
        if os.path.exists(saved_pvals_name):
            p_val_cal_fold_dict = data.load_cached_data(saved_pvals_name)
        else:
            p_val_cal_fold_dict = scores.compute_p_values_cred_and_conf(
                _probs=_train_probs,
                # simls_b=_train_simls_b,
                # simls_m=_train_simls_m,
                y_true=_train_y_true,
                _test_probs=_cal_probs,
                # test_simls_b=_cal_simls_b,
                # test_simls_m=_cal_simls_m,
                y_test=_cal_y_true)
            data.cache_data(p_val_cal_fold_dict, saved_pvals_name)
    else:
        p_val_cal_fold_dict = scores.compute_p_values_cred_and_conf(
            _probs=_train_probs,
            # simls_b=_train_simls_b,
            # simls_m=_train_simls_m,
            y_true=_train_y_true,
            _test_probs=_cal_probs,
            # test_simls_b=_cal_simls_b,
            # test_simls_m=_cal_simls_m,
            y_test=_cal_y_true)
    # print('p_val_cal_fold_dict: {}'.format(p_val_cal_fold_dict))
    cred_p_val_cal = p_val_cal_fold_dict['cred']
    conf_p_val_cal = p_val_cal_fold_dict['conf']

    # ---------------------------------------- #
    # 2. Find Calibration Thresholds           #
    # ---------------------------------------- #
    if args.thresholds == 'quartiles':
        if 'cred' in args.criteria:
            logging.info('Finding cred p-value thresholds (quartiles)...')
            cred_p_val_thresholds = thresholding.find_quartile_thresholds(
                scores=cred_p_val_cal,
                predicted_labels=_cal_y_pred,
                groundtruth_labels=_cal_y_true,
                consider=args.q_consider)

        if 'conf' in args.criteria:
            # logging.info('Finding conf p-value thresholds (quartiles)...')
            conf_p_val_thresholds = thresholding.find_quartile_thresholds(
                scores=conf_p_val_cal,
                predicted_labels=_cal_y_pred,
                groundtruth_labels=_cal_y_true,
                consider=args.q_consider)
    else:
        msg = 'Unknown option: args.thresholds = {}'.format(args.threshold)
        logging.critical(msg)
        raise ValueError(msg)

    # ---------------------------------------- #
    # 4. Score and Predict Test Observations   #
    # ---------------------------------------- #
    if tostore:
        saved_pvals_name = 'p_val_test_dict.p'
        saved_pvals_name = os.path.join(_saved_data_folder, saved_pvals_name)
        if os.path.exists(saved_pvals_name):
            p_val_test_dict = data.load_cached_data(saved_pvals_name)
        else:
            p_val_test_dict = scores.compute_p_values_cred_and_conf(
                _probs=_cal_probs,
                y_true=_cal_y_true,
                _test_probs=_test_probs,
                y_test=_test_y_pred)
            data.cache_data(p_val_test_dict, saved_pvals_name)
    else:
        p_val_test_dict = scores.compute_p_values_cred_and_conf(
            _probs=_cal_probs,
            y_true=_cal_y_true,
            _test_probs=_test_probs,
            y_test=_test_y_pred)
    # ---------------------------------------- #
    # 5. Apply Thresholds, Compare Results     #
    # ---------------------------------------- #
    report_str = ''

    def print_and_extend(report_line):
        nonlocal report_str
        cprint(report_line, 'red')
        report_str += report_line + '\n'

    if args.thresholds == 'quartiles':
        # for q in ('q1', 'q2', 'q3', 'mean'):
        # for q in ('aus1', 'aus2', 'aus3',):
        for q in ('aus1',):
            p_val_binary_thresholds = {}

            if 'cred' in args.criteria:
                p_val_binary_thresholds['cred'] = cred_p_val_thresholds[q]
            if 'conf' in args.criteria:
                # print('CONF HERE!')
                # print(args.criteria)
                p_val_binary_thresholds['conf'] = conf_p_val_thresholds[q]

            print_and_extend('=' * 40)
            print_and_extend('[P-VALS] Threshold criteria: {}'.format(q))
            report_str += print_thresholds(p_val_binary_thresholds)

            # keep_mask = thresholding.apply_threshold(
            #     binary_thresholds=p_val_binary_thresholds,
            #     test_scores=p_val_test_dict,
            #     y_test=_test_y_pred)

            results, keep_mask, tttt = thresholding.test_with_rejection(
                multi_thresholds=p_val_binary_thresholds,
                test_scores=p_val_test_dict,
                groundtruth_labels=_test_y_true,
                predicted_labels=_test_y_pred,
                full=False)
            report_str += thresholding.report_results(d=results)
            # store_keep_file(_test_simls_b, _test_simls_m, p_val_test_dict, _test_y_true, _test_y_pred,
            #                 keep_mask, _saved_data_folder, q)

    elif args.thresholds in ('random-search', 'constrained-search'):

        print_and_extend('=' * 40)
        print_and_extend('[P-VALS] Threshold with random grid search')
        report_str += print_thresholds(p_val_found_thresholds)

        # keep_mask = thresholding.apply_threshold(
        #     binary_thresholds=p_val_found_thresholds,
        #     test_scores=p_val_test_dict,
        #     y_test=_test_y_pred)

        results, keep_mask, tttt = thresholding.test_with_rejection(
            multi_thresholds=p_val_found_thresholds,
            test_scores=p_val_test_dict,
            groundtruth_labels=_test_y_true,
            predicted_labels=_test_y_pred)
        report_str += thresholding.report_results(d=results)
        # store_keep_file(_test_simls_b, _test_simls_m, p_val_test_dict, _test_y_true, _test_y_pred,
        #                 keep_mask, _saved_data_folder, 'constr')

    else:
        raise ValueError(
            'Unknown option: args.thresholds = {}'.format(args.threshold))

    # with open(os.path.join(_saved_data_folder, 'keep_mask.csv'), 'w+', newline='') as keep_mask_writer:
    #     keep_mask_writer.write(keep_mask_str)
    # # -------------- result-------------------
    # with open(os.path.join(_saved_data_folder, 'result.txt'), 'w+') as result_writer:
    #     result_writer.write(report_str)

    end_time = datetime.datetime.now()
    print('结束计时：{}'.format(end_time))
    time_space = end_time - start_time
    print('总计时：{}'.format(time_space))
    keep_num=sum(keep_mask)
    reject_num=len(keep_mask)-keep_num
    reject_rate = float(reject_num) / (keep_num + reject_num)
    # keep_masking = pd.DataFrame(keep_mask)
    # keep_masking.to_csv("keep_mask", index=False)
    # print(sum(keep_mask))

    # anom_cred = np.array(p_val_test_dict['cred'])[~keep_mask]
    # anom_conf = np.array(p_val_test_dict['conf'])[~keep_mask]
    # anom_score = [2 - x - y for x, y in zip(anom_cred, anom_conf)]
    #
    # order_idx = sorted(range(len(anom_score)), key=lambda i: anom_score[i], reverse=True)
    X_anom_cred = np.array(p_val_test_dict['cred'])
    X_anom_conf = np.array(p_val_test_dict['conf'])
    X_anom_score = [2 - x - y for x, y in zip(X_anom_cred, X_anom_conf)]

    order_idx = sorted(range(len(X_anom_score)), key=lambda i: X_anom_score[i], reverse=True)

    # return keep_mask, rate, order_idx, anom_score
    return keep_mask, reject_rate, order_idx, X_anom_score


def reject(dataset, keep_mask, test_y_predict):
    dataset.reset_index(drop=True, inplace=True)
    keep_mask = pd.DataFrame(keep_mask, columns=['0'])
    test_y_predict = pd.DataFrame(test_y_predict, columns=['y_pred'])
    # test_y_predict = pd.DataFrame(test_y_predict, columns=['Classification'])
    # dataset[' Label'] = test_y_predict[' Label']
    keep_mask = keep_mask['0']
    if len(dataset) != len(keep_mask):
        print('index error')

    re = dataset.loc[~keep_mask,]
    # re[' Label'] = 1 - re[' Label']
    # re1 = re.loc[re[' Label'] == 1]
    # re0 = re.loc[re[' Label'] == 0]
    ke = dataset.loc[keep_mask,]
    ke_pred = test_y_predict.loc[keep_mask,]
    re_pred = test_y_predict.loc[~keep_mask,]
    # ================================获取keep部分标签=====================================

    # ke[' Label'] = ke_pred[' Label']
    # ===================================================================================
    # ke['Classification'] = ke_pred['Classification']
    # ke1 = ke.loc[ke[' Label'] == 1]
    # ke0 = ke.loc[ke[' Label'] == 0]
    # filename = f"reject_{i}.csv"
    # re.to_csv(filename, index=False)
    # re_pred.to_csv("./data/rwdids/reject_pred.csv", index=False)
    # return re1, re0, ke1, ke0
    return re, ke, re_pred, ke_pred


# def store_keep_file(_test_simls_b, _test_simls_m, p_val_test_dict, _test_y_true, _test_y_pred,
#                     keep_mask, _saved_data_folder, descript=''):
#     X_keep = []
#     X_reject = []
#     X_keep.append(
#         ['test_siml_b'] + ['test_siml_m'] + ['test_cred'] + ['test_conf'] + ['y_true'] + ['y_pred'] + ['index'])
#     X_reject.append(
#         ['test_siml_b'] + ['test_siml_m'] + ['test_cred'] + ['test_conf'] + ['y_true'] + ['y_pred'] + ['index'])
#
#     big_array = np.array(([_test_simls_b])).T
#     big_array = np.hstack((big_array, np.array(([_test_simls_m])).T))
#     big_array = np.hstack((big_array, np.array(([p_val_test_dict['cred']])).T))
#     big_array = np.hstack((big_array, np.array(([p_val_test_dict['conf']])).T))
#     big_array = np.hstack((big_array, np.array(([_test_y_true])).T))
#     big_array = np.hstack((big_array, np.array(([_test_y_pred])).T))
#     for i in range(len(keep_mask)):
#         array_line = list(big_array[i])
#         array_line.append(i)
#
#         if keep_mask[i]:
#             X_keep.append(array_line)
#         elif not keep_mask[i]:
#             X_reject.append(array_line)
#         else:
#             print('error big_array')
#
#     # if descript != '':
#     #     descript += '_'
#     # save keep/reject as file
#     # with open(os.path.join(_saved_data_folder, descript + 'Xy_keep.csv'), 'w+', newline='') as csv_file2:
#     #     f_csv_writer = csv.writer(csv_file2)
#     #     f_csv_writer.writerows(X_keep)
#     #     csv_file2.close()
#     # with open(os.path.join(_saved_data_folder, descript + 'Xy_reject.csv'), 'w+', newline='') as csv_file2:
#     #     f_csv_writer = csv.writer(csv_file2)
#     #     f_csv_writer.writerows(X_reject)
#     #     csv_file2.close()


# 打包，方便score评价时取用
def package_cred_conf(cred_values, conf_values, criteria):
    package = {}
    if 'cred' in criteria:
        package['cred'] = cred_values
    if 'conf' in criteria:
        package['conf'] = conf_values
    return package


# 将svm的siml[]转换为siml[[],[]]  # 因为SVM的NCM是单个的，这里预处理为两个
def convert_ncm_to_siml(__train_ncms_x, __train_y_true):
    __train_simls_b = []
    __train_simls_m = []
    for single_n, single_y in zip(__train_ncms_x, __train_y_true):
        if single_y == 1:
            __train_simls_b.append(single_n[0])
            __train_simls_m.append(-single_n[0])
        elif single_y == 0:
            __train_simls_b.append(-single_n[0])
            __train_simls_m.append(single_n[0])
        else:
            print('siml convert error train!!!')
            input()
    return __train_simls_b, __train_simls_m


def print_thresholds(binary_thresholds):
    # Display per-class thresholds
    s = ''

    def out_thresholds(a_dict):
        out_str = ''
        keys = a_dict.keys()
        for i_key in keys:
            out_str += ' {} = {:.6f},'.format(i_key, a_dict[i_key])
        return out_str

    if 'cred' in binary_thresholds:
        s = ('Cred thresholds:' + out_thresholds(binary_thresholds['cred']))

        # s = ('Cred thresholds: mw {:.6f}, gw {:.6f}'.format(
        #     binary_thresholds['cred']['mw'],
        #     binary_thresholds['cred']['gw']))
    if 'conf' in binary_thresholds:
        s = ('\n\tConf thresholds:' + out_thresholds(binary_thresholds['conf']))

        # s += ('\n\tConf thresholds: mw {:.6f}, gw {:.6f}'.format(
        #     binary_thresholds['conf']['mw'],
        #     binary_thresholds['conf']['gw']))
    if 'cred' not in binary_thresholds and 'conf' not in binary_thresholds:
        s = 'No threshold found!'
    logging.info(s)
    return s


# def read_csv(filename, signal='NO'):
#     csv_file1 = open(os.path.join('features/', filename))
#     f_csv_reader = csv.reader(csv_file1)
#     total_features = []
#     for csv_row in f_csv_reader:
#         if f_csv_reader.line_num != 1:
#             if signal == 'int':
#                 total_features.append(int(csv_row[1]))
#             else:
#                 total_features.append(float(csv_row[1]))
#
#     return total_features


# def main():
#     # import csv过滤NaN as aaa
#     # aaa.main()

#     description = ''
#     from my_tool import to_csv, read_csv

#     train_probs = read_csv('train_probs.csv')
#     train_y_true = read_csv('train_y_true.csv').astype(int)
#     cal_probs = read_csv('cal_probs.csv')
#     cal_y_true = read_csv('cal_y_true.csv').astype(int)
#     cal_y_pred = read_csv('cal_y_predict.csv').astype(int)
#     test_probs = read_csv('test_probs.csv')
#     test_y_true = read_csv('test_y.csv').astype(int)
#     test_y_pred = read_csv('test_y_predict.csv').astype(int)

#     start_half_transcend(train_probs, train_y_true,
#                          cal_probs, cal_y_true, cal_y_pred,
#                          test_probs, test_y_true, test_y_pred,
#                          _saved_data_folder=os.path.join('half_transcend_ce\\models', 'half-ce-{}'.format(description)),
#                          tostore=True)


# if __name__ == '__main__':
#     main()

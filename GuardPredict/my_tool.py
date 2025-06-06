import os, pickle, csv
import numpy as np
from timeit import default_timer as timer
from sklearn import metrics as metrics
import shutil
from tqdm import tqdm
import random

def now():
    from datetime import datetime
    return datetime.now()


def del_dir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def format_path(data_path, Itype='.p'):
    '''补充后缀如.p生成对应目录避免错误'''
    if '.' not in data_path or data_path.split('.')[-1] != Itype[1:]:
        print('后缀有误，已自动添加{}'.format(Itype))
        data_path += Itype
    folder_path = os.path.dirname(data_path)
    if folder_path != '' and not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return data_path


def cache_data(data, data_path):
    '''.p存'''
    data_path = format_path(data_path, Itype='.p')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print('Done cache_data for {}.'.format(data_path))


def load_cached_data(data_path):
    '''.p读'''
    data_path = format_path(data_path, Itype='.p')

    with open(data_path, 'rb') as f:
        model = pickle.load(f)
    print('Done load_cached_data for {}.'.format(data_path))
    return model


def to_csv(pred, csv_path):
    '''to_csv与read_csv配套，用于list或者是nparray'''
    csv_path = format_path(csv_path, Itype='.csv')
    if (type(pred[0]) != type([])) and (type(pred[0]) != type(np.array([]))):
        pred = np.array(pred).reshape(-1, 1)
    # pred = np.array(pred).reshape(-1, col_num)

    with open(csv_path, 'w', newline='') as f:
        f_csv_writer = csv.writer(f)

        f_csv_writer.writerows(pred)
        f.close()


def read_csv(csv_path):
    '''to_csv与read_csv配套，用于list或者是nparray'''
    csv_path = format_path(csv_path, Itype='.csv')
    RMSEs = []
    one_array_flag = True
    with open(csv_path, 'r') as f:
        for csv_row in f:
            i_RMSEs = []
            for i_data in csv_row.split(','):
                i_RMSEs.append(float(i_data))
            if one_array_flag:
                if len(i_RMSEs) == 1:
                    RMSEs.extend(i_RMSEs)
                else:
                    one_array_flag = False
                    new_RMSEs = []
                    for q in RMSEs:
                        new_RMSEs.append(np.array([q]))
                    RMSEs = new_RMSEs
            if not one_array_flag:
                RMSEs.append(np.array(i_RMSEs))
        f.close()
    return np.array(RMSEs)


def set_random_seed(seed=42, deterministic=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass
def evaluate_true_pred_label(y_true, y_pred, desc='', para='strong'):
    '''根据true和pred评估结果，true在前'''
    y_true = np.array([0 if xxx == 0 else 1 for xxx in y_true])
    y_pred = np.array([0 if xxx == 0 else 1 for xxx in y_pred])
    num = y_true.shape[0]
    if num == 0:
        return

    print('-' * 10 + desc + '-' * 10)
    cf_flow = metrics.confusion_matrix(y_true, y_pred)
    if len(cf_flow.ravel()) == 1:
        if y_true[0] == 0:
            TN, FP, FN, TP = cf_flow[0][0], 0, 0, 0
        elif y_true[0] == 1:
            TN, FP, FN, TP = 0, 0, 0, cf_flow[0][0]
        else:
            raise Exception("label error")
    else:
        TN, FP, FN, TP = cf_flow.ravel()

    rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
    Accu = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0

    F1 = 2 * rec * prec / (rec + prec) if (rec + prec) != 0 else 0
    if para.lower() == 'strong'.lower():
        print("TP:\t" + str(TP), end='\t|| ')
        print("FP:\t" + str(FP), end='\t|| ')
        print("TN:\t" + str(TN), end='\t|| ')
        print("FN:\t" + str(FN))
        print("Recall:\t{:6.4f}".format(rec), end='\t|| ')
        print("Precision:\t{:6.4f}".format(prec))
        print("Accuracy:\t{:6.4f}".format(Accu), end='\t|| ')
        print("F1:\t{:6.4f}".format(F1))
    else:
        print("\tTP \t" + str(TP), end='\t - ')
        print("\tFP \t" + str(FP), end='\t - ')
        print("\tTN \t" + str(TN), end='\t - ')
        print("\tFN \t" + str(FN))
        print("\tRecall \t{:6.4f}".format(rec), end='\t - ')
        print("\tPrecision \t{:6.4f}".format(prec))
        print("\tAccuracy \t{:6.4f}".format(Accu), end='\t - ')
        print("\tF1 \t{:6.4f}".format(F1))
    info = [desc,TP,FP,TN,FN,rec,prec,Accu,F1]
    return info

def search_for_OWAD_thres(rmse, y_true, thres, epoch=0):
    y_pred = []
    for i_r in rmse:
        if i_r < thres:
            y_pred.append(0)
        else:
            y_pred.append(1)

    cf_flow = metrics.confusion_matrix(y_true, y_pred)
    if len(cf_flow.ravel()) == 1:
        if y_true[0] == 0:
            TN, FP, FN, TP = cf_flow[0][0], 0, 0, 0
        elif y_true[0] == 1:
            TN, FP, FN, TP = 0, 0, 0, cf_flow[0][0]
        else:
            raise Exception("label error")
    else:
        TN, FP, FN, TP = cf_flow.ravel()
    rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
    score2 = (TN / (TN + FP)) if (TN + FP) != 0 else 0

    if np.abs(rec - score2) < 0.1 + epoch * 0.01 and np.abs(rec - prec) < 0.1 + epoch * 0.01:
        print('')
        return thres
    elif rec > score2:
        print(end='↑')
        return search_for_OWAD_thres(rmse, y_true, thres * 11 / 9, epoch + 1)
    else:
        print(end='↓')
        return search_for_OWAD_thres(rmse, y_true, thres * 8 / 9, epoch + 1)


def read_csv_temp(paths):
    full_X, full_y = [], []
    # 提取全部X,y,t
    for i_path in paths:
        if os.path.exists('./MY_general_tools/##tempXy_cache/' + i_path.split('/')[-1] + '.p'):
            data = load_cached_data('./MY_general_tools/##tempXy_cache/' + i_path.split('/')[-1] + '.p')
            tempX, tempy = data['tempX'], data['tempy']
        else:
            tempX, tempy = [], []
            with open(i_path + '.csv', 'r') as f_csv:
                next(f_csv)
                for line in tqdm(f_csv):
                    line = line.strip().split(',')
                    single_X = np.array(line[:-1]).astype(float)
                    single_y = np.array(line[-1]).astype(int)
                    tempX.append(single_X)
                    tempy.append(single_y)
            tempX = np.array(tempX)
            tempy = np.array(tempy)
            data = {'tempX': tempX, 'tempy': tempy}
            cache_data(data, './MY_general_tools/##tempXy_cache/' + i_path.split('/')[-1] + '.p')

        print(len(tempy))
        # data_flag = None
        # for i_item in random_size:
        #     if i_item in paths[0]:
        #         data_flag = i_item

        # if data_flag is None:
        #     for i in range(6):
        #         print('Not Knowing DataSet, random opt. canceled.')
        # else:
        #     st_at, ed_at = idx_at[data_flag][path_cnt]
        #     X_idx = np.arange(ed_at - st_at) + st_at
        #     np.random.shuffle(X_idx)
        #     if path_cnt == 0:
        #         X_idx = np.array(sorted(X_idx[:setting['Train_Num']]))
        #     else:
        #         X_idx = np.array(sorted(X_idx[:random_size[data_flag]]))
        #     new_tempX = tempX[X_idx]
        #     new_tempy = tempy[X_idx]
        #     path_cnt += 1
        #     while len(tempy) > ed_at:
        #         st_at, ed_at = idx_at[data_flag][path_cnt]
        #         X_idx = np.arange(ed_at - st_at) + st_at
        #         np.random.shuffle(X_idx)

        #         X_idx = np.array(sorted(X_idx[:random_size[data_flag]]))
        #         new_tempX = np.concatenate((new_tempX, tempX[X_idx]))
        #         new_tempy = np.concatenate((new_tempy, tempy[X_idx]))
        #         path_cnt += 1
        #     tempX, tempy = new_tempX, new_tempy

        full_X.extend(tempX)
        full_y.extend(tempy)
        print(len(tempX), sum(tempy), len(tempX[0]))
        del tempX, tempy
    full_X = np.array(full_X)
    full_y = np.array(full_y)
    return full_X, full_y


def load_pd_csv(paths=['E:/99论文/ENIDrift-main/data/extracted_sematic_drift_1to0'],
                rand_flag=False,
                formats='.csv',
                cache_root='../MY_general_tools/##tempXy_cache/'):
    if rand_flag:
        # 只适配kyoto
        data_flag = '2006'
        if data_flag not in paths[0]:
            import sys
            sys.exit(-2)
    print(f'Loading dataset by pandas...')
    w = 10000
    full_X, full_y = None, None
    # 提取全部X,y,t
    for i_path in paths:
        if os.path.exists(cache_root + i_path.split('/')[-1] + '.p'):
            data = load_cached_data(cache_root + i_path.split('/')[-1] + '.p')
            tempX, tempy = data['tempX'], data['tempy']
            try:
                print('total len:{}, sum = {}, feature_size = {}'.format(len(tempy), sum(tempy), len(tempX[0])))
            except:
                print('total len:{}, feature_size = {}'.format(len(tempy), len(tempX[0])))
        else:
            path = format_path(i_path, formats)
            import pandas as pd
            source = pd.read_csv(path, index_col=None)

            ##打乱source操作
            source.index = [random.choice(range(len(source))) for _ in range(len(source))]

            width = source.shape[1]
            if '202006101400' in i_path:
                tempX = source.iloc[:200 * w, :].values
                tempy = np.array([0] * len(tempX))
                tempX = list(tempX)
                print('[*MAWI*]Searching lower char...')
                for i in tqdm(range(len(tempX))):
                    for j in range(len(tempX[0])):
                        if type(tempX[i][j]) == type('') and '.' not in tempX[i][j] and ':' not in tempX[i][j]:
                            try:
                                tttttt = int(tempX[i][j])
                            except:
                                tempX[i][j] = tempX[i][j].lower()
                tempX = np.array(tempX)
                print('[*MAWI*]Done searching lower char...')
            elif '1RW' in i_path:
                tempX, tempy = source.iloc[:, :width - 1].values, source.iloc[:, width - 1].values
                print('[*1RW*]Searching lower char...')
                for i in tqdm(range(len(tempX))):
                    for j in range(len(tempX[0])):
                        if type(tempX[i][j]) == type('') and '.' not in tempX[i][j] and ':' not in tempX[i][j]:
                            try:
                                tttttt = int(tempX[i][j])
                            except:
                                tempX[i][j] = tempX[i][j].lower()
                print('[*1RW*]Done searching lower char...')
            elif '3ML' in i_path:
                tempX, tempy = source.iloc[:, :width - 1].values, source.iloc[:, width - 1].values
                tempX = list(tempX)
                print('[*3ML*]Appending idx=6,7...')
                for i in tqdm(range(len(tempX))):
                    new_line = list(np.arange(width - 1 - 1))
                    new_line[:6] = tempX[i][:6]
                    new_line[6] = '{}_{}'.format(tempX[i][6], tempX[i][7])
                    new_line[7:] = tempX[i][8:]
                    tempX[i] = np.array(new_line)
                tempX = np.array(tempX)
                print('[*3ML*]Done')

            else:
                tempX, tempy = source.iloc[:, :width - 1].values, source.iloc[:, width - 1].values
                try:
                    print('total len:{}, sum = {}, feature_size = {}'.format(len(tempy), sum(tempy), len(tempX[0])))
                except:
                    print('total len:{}, feature_size = {}'.format(len(tempy), len(tempX[0])))

            data = {'tempX': tempX, 'tempy': tempy}
            cache_data(data, cache_root + i_path.split('/')[-1] + '.p')
        if rand_flag:
            st_at, ed_at = [0 * w, 40 * w]
            X_idx = np.arange(ed_at - st_at) + st_at
            X_idx = np.array(sorted(X_idx[:10 * w]))

            tempX = tempX[X_idx]
            tempy = tempy[X_idx]
        if full_X is None:
            full_X = tempX
            full_y = tempy
        else:
            full_X = np.concatenate((full_X, tempX))
            full_y = np.concatenate((full_y, tempy))
    print(f'Finnished load dataset by pandas.')
    return full_X, full_y


def mlp_calculate_ncm(train_X, cal_X, X_test, model):
    # import torch
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to('cpu')
    train_prob = model.predict_proba(train_X)
    # train_b = train_prob[:, 0]
    # train_m = train_prob[:, 1]
    cal_prob = model.predict_proba(cal_X)
    # cal_b = cal_prob[:, 0]
    # cal_m = cal_prob[:, 1]
    cal_y_pred = model.predict(cal_X)

    test_prob = model.predict_proba(X_test)
    # test_b = test_prob[:, 0]
    # test_m = test_prob[:, 1]
    test_y_pred = model.predict(X_test)

    return train_prob, cal_prob, cal_y_pred, test_prob, test_y_pred


if __name__ == '__main__':
    pass
    y_true = [3, 0]
    y_pred = [2, 3]
    evaluate_true_pred_label(y_true, y_pred, desc='')

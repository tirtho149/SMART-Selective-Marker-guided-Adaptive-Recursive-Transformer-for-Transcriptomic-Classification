import datetime
import logging
import os
import sys
from copy import deepcopy
from os import makedirs
from os.path import join, exists
from posixpath import abspath

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from data.data_access import Data
from model.model_factory import get_model
from pipeline.one_split import OneSplitPipeline
from utils.plots import plot_box_plot
from utils.rnd import set_random_seeds

# --- bioMOR shared contract: identical seed-42 CV5 folds + common score CSV ---
_PNET_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # baseline_pnet
_REPO_ROOT = os.path.dirname(_PNET_BASE)                                  # repo root
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import biomor_common as bc  # noqa: E402

timeStamp = '_{0:%b}-{0:%d}_{0:%H}-{0:%M}'.format(datetime.datetime.now())


def save_model(model, model_name, directory_name):
    filename = join(abspath(directory_name), 'fs')
    logging.info('saving model {} coef to dir ({})'.format(model_name, filename))
    if not exists(filename.strip()):
        makedirs(filename)
    filename = join(filename, model_name + '.h5')
    logging.info('FS dir ({})'.format(filename))
    model.save_model(filename)


def get_mean_variance(scores):
    df = pd.DataFrame(scores)
    return df, df.mean(), df.std()


class CrossvalidationPipeline(OneSplitPipeline):
    def __init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params, exp_name):
        OneSplitPipeline.__init__(self, task, data_params, pre_params, feature_params, model_params, pipeline_params,
                                  exp_name)

    def run(self, n_splits=5):

        list_model_scores = []
        model_names = []

        for data_params in self.data_params:
            data_id = data_params['id']
            # logging
            logging.info('loading data....')
            data = Data(**data_params)

            # bioMOR: use the FULL cohort (train+val+test) in the reader's
            # patient-sorted order, then split with bc.cv_folds so folds are
            # byte-identical to bioMoR's seed-42 CV5. Per-fold CNV z-scoring is
            # applied inside train_predict_crossvalidation (the reader's own
            # z-score is disabled via zscore_cnv=False in the param file).
            X, y, info, cols = data.get_data()
            y = np.asarray(y).ravel()
            info = np.asarray(info)

            # get model
            logging.info('fitting model ...')

            for model_param in self.model_params:
                if 'id' in model_param:
                    model_name = model_param['id']
                else:
                    model_name = model_param['type']

                set_random_seeds(random_seed=20080808)
                model_name = model_name + '_' + data_id
                m_param = deepcopy(model_param)
                m_param['id'] = model_name
                logging.info('fitting model ...')

                scores = self.train_predict_crossvalidation(m_param, X, y, info, cols, model_name)
                scores_df, scores_mean, scores_std = get_mean_variance(scores)
                list_model_scores.append(scores_df)
                model_names.append(model_name)
                self.save_score(data_params, m_param, scores_df, scores_mean, scores_std, model_name)
                logging.info('scores')
                logging.info(scores_df)
                logging.info('mean')
                logging.info(scores_mean)
                logging.info('std')
                logging.info(scores_std)

        df = pd.concat(list_model_scores, axis=1, keys=model_names)
        df.to_csv(join(self.directory, 'folds.csv'))
        plot_box_plot(df, self.directory)

        return scores_df

    def save_prediction(self, info, y_pred, y_pred_score, y_test, fold_num, model_name, training=False):
        if training:
            file_name = join(self.directory, model_name + '_traing_fold_' + str(fold_num) + '.csv')
        else:
            file_name = join(self.directory, model_name + '_testing_fold_' + str(fold_num) + '.csv')
        logging.info("saving : %s" % file_name)
        info['pred'] = np.asarray(y_pred).ravel()
        score_arr = np.asarray(y_pred_score)
        if score_arr.ndim == 2 and score_arr.shape[1] > 1:
            # Multiclass: one column per class (pred_score_0 .. pred_score_K-1)
            for k in range(score_arr.shape[1]):
                info['pred_score_%d' % k] = score_arr[:, k]
        else:
            info['pred_score'] = score_arr.ravel()
        info['y'] = np.asarray(y_test).ravel()
        info.to_csv(file_name)

    def _zscore_cnv_per_fold(self, x_train, x_test):
        """Z-score the CNV (odd) columns of the gene-grouped [mut,cnv] matrix
        using TRAIN statistics only. Mutation (even) columns stay binary."""
        cnv_cols = np.arange(1, x_train.shape[1], 2)
        mu = x_train[:, cnv_cols].mean(axis=0, keepdims=True)
        sd = x_train[:, cnv_cols].std(axis=0, keepdims=True) + 1e-8
        x_train = x_train.copy()
        x_test = x_test.copy()
        x_train[:, cnv_cols] = (x_train[:, cnv_cols] - mu) / sd
        x_test[:, cnv_cols] = (x_test[:, cnv_cols] - mu) / sd
        return x_train, x_test

    def train_predict_crossvalidation(self, model_params, X, y, info, cols, model_name):
        logging.info('model_params: {}'.format(model_params))
        # bioMOR: identical seed-42 CV5 (train/val/test) folds.
        folds = bc.cv_folds(y)
        if os.environ.get('PNET_SMOKE', '0') == '1':
            folds = folds[:1]
            # Shrink epochs for a fast smoke check.
            try:
                model_params['params']['fitting_params']['epoch'] = 3
            except Exception:
                pass
        i = 0
        scores = []
        model_list = []
        bc_f1, bc_acc, bc_nt = [], [], []
        for train_index, val_index, test_index in folds:
            model = get_model(model_params)
            logging.info('fold # ----------------%d---------' % i)
            x_train_all = X[np.concatenate([train_index, val_index])]
            x_test = X[test_index]
            y_train_all = y[np.concatenate([train_index, val_index])]
            y_test = y[test_index]
            info_test = pd.DataFrame(index=info[test_index])
            info_train = pd.DataFrame(index=info[np.concatenate([train_index, val_index])])

            # Per-fold CNV z-score (train stats only) on the gene-grouped matrix.
            x_train_all, x_test = self._zscore_cnv_per_fold(x_train_all, x_test)
            x_train_all, x_test = self.preprocess(x_train_all, x_test)
            x_train_all, x_test = self.extract_features(x_train_all, x_test)

            # Inner train/val (bc's own within-train val split); z-score val
            # using the inner-train CNV stats only.
            x_tr, x_val = self._zscore_cnv_per_fold(X[train_index], X[val_index])
            y_tr = y[train_index]
            y_val = y[val_index]
            model = model.fit(x_tr, y_tr, x_val, y_val)
            x_train = x_train_all
            y_train = y_train_all

            y_pred_test, y_pred_test_scores = self.predict(model, x_test, y_test)
            score_test = self.evaluate(y_test, y_pred_test, y_pred_test_scores)
            logging.info('model {} -- Test score {}'.format(model_name, score_test))
            self.save_prediction(info_test, y_pred_test, y_pred_test_scores, y_test, i, model_name)

            if hasattr(model, 'save_model'):
                logging.info('saving coef')
                save_model(model, model_name + '_' + str(i), self.directory)

            if self.save_train:
                logging.info('predicting training ...')
                y_pred_train, y_pred_train_scores = self.predict(model, x_train, y_train)
                self.save_prediction(info_train, y_pred_train, y_pred_train_scores, y_train, i, model_name,
                                     training=True)

            scores.append(score_test)
            bc_f1.append(100.0 * float(score_test.get('f1_macro', 0.0)))
            bc_acc.append(100.0 * float(score_test.get('accuracy', 0.0)))
            bc_nt.append(int(len(test_index)))

            fs_parmas = deepcopy(model_params)
            if hasattr(fs_parmas, 'id'):
                fs_parmas['id'] = fs_parmas['id'] + '_fold_' + str(i)
            else:
                fs_parmas['id'] = fs_parmas['type'] + '_fold_' + str(i)

            model_list.append((model, fs_parmas))
            i += 1
        self.save_coef(model_list, cols)
        logging.info(scores)

        # bioMOR common-schema score CSV (dataset,model,fold,macro_f1,accuracy,n_test).
        try:
            cohort = os.environ.get('PNET_COHORT', str(model_name))
            work_dir = os.path.join(_PNET_BASE, 'work_dirs', cohort)
            out = bc.write_scores(work_dir, model='P-NET', dataset=cohort,
                                  fold_f1=bc_f1, fold_acc=bc_acc, fold_ntest=bc_nt)
            logging.info('bioMOR scores written to %s', out)
            print('[pnet] bioMOR scores ->', out, 'mean macro-F1=%.2f' % (np.mean(bc_f1)))
        except Exception as e:
            logging.warning('bc.write_scores failed: %s', e)
        return scores

    def save_score(self, data_params, model_params, scores, scores_mean, scores_std, model_name):
        file_name = join(self.directory, model_name + '_params' + '.yml')
        logging.info("saving yml : %s" % file_name)
        with open(file_name, 'w') as yaml_file:
            yaml_file.write(
                yaml.dump({'data': data_params, 'models': model_params, 'pre': self.pre_params,
                           'pipeline': self.pipeline_params, 'scores': scores.to_json(),
                           'scores_mean': scores_mean.to_json(), 'scores_std': scores_std.to_json()},
                          default_flow_style=False))

        # Human-readable CSV: per-fold rows + mean/std summary at the bottom.
        summary_csv = join(self.directory, model_name + '_scores.csv')
        summary = scores.copy()
        summary.index = ['fold_%d' % i for i in range(len(summary))]
        summary.loc['mean'] = scores_mean
        summary.loc['std'] = scores_std
        summary.to_csv(summary_csv)
        logging.info("saving cv summary csv : %s" % summary_csv)

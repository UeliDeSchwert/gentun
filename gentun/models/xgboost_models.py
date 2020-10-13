#!/usr/bin/env python
"""
Machine Learning models compatible with the Genetic Algorithm implemented using xgboost
"""

import xgboost as xgb
import numpy as np

from .generic_models import GentunModel


class XgboostModel(GentunModel):

    def __init__(self, x_train, y_train, hyperparameters, booster='gbtree', objective='reg:linear',
                 eval_metric='rmse', kfold=5, num_boost_round=5000, early_stopping_rounds=100):
        super(XgboostModel, self).__init__(x_train, y_train)
        self.params = {
            'booster': booster,
            'objective': objective,
            'eval_metric': eval_metric,
            'silent': 1
        }
        self.params.update(hyperparameters)
        self.eval_metric = eval_metric
        self.kfold = kfold
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds

    def cross_validate(self):
        """Train model using k-fold cross validation and
        return mean value of validation metric.
        """
        d_train = xgb.DMatrix(self.x_train, label=self.y_train)
        # xgb calls its k-fold cross-validation parameter 'nfold'
        cv_result = xgb.cv(
            self.params, d_train, num_boost_round=self.num_boost_round,
            early_stopping_rounds=self.early_stopping_rounds, nfold=self.kfold
        )

        return np.mean(cv_result['test-{}-mean'.format(self.eval_metric)])

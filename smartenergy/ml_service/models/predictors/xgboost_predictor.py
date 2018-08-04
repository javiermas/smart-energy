"""XGBoost predictor super class"""

import logging
import pandas as pd
from numpy import mean
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import _pickle as cPickle
from .base import Predictor


class XGBoostPredictor(Predictor):

    def __init__(self, hyperparameters, **kwargs):
        super().__init__(hyperparameters, **kwargs)
        self.Model = XGBRegressor
        self.model = None

    @Predictor.apply_schemata
    def train(self, features, target):
        logging.info('Fitting {} model...'.format(self.__class__.__name__))
        self.model = self.Model(**self.hyperparameters)
        self.model.fit(features, target)
        logging.info('Model successfully fitted')

    @Predictor.apply_schemata
    def predict(self, features):
        logging.info('Generating {} model prediction...'.format(self.__class__.__name__))
        features = features[self.model.get_booster().feature_names]
        prediction = pd.DataFrame(self.model.predict(features),
                                  index=features.index,
                                  columns=self.target_schema.keys())
        logging.info('Model prediction successfully generated')
        return prediction

    @Predictor.apply_schemata
    def validate(self, features, target):
        metrics = []
        for tr_i, t_i in KFold(n_splits=5, shuffle=True).split(features, target):
            train_data = features.iloc[tr_i], target.iloc[tr_i].squeeze()
            test_features, test_target = features.iloc[t_i], target.iloc[t_i]
            cv_model = self.Model(**self.hyperparameters)
            cv_model.fit(*train_data)
            pred = cv_model.predict(test_features)
            metrics.append(mae(test_target, pred, [mean(test_target)] * len(pred)))

        result = pd.concat([pd.Series(metrics).add_prefix('fold_')
                            .add_suffix('_metric'), pd.Series(self.hyperparameters)])
        result['mean_metric'] = result.loc[[c for c in result.index if c[:5] == 'fold_']].mean()
        return {'mae': round(result['mean_metric'], 2)}

    def serialize(self, stream):
        logging.info('Dumping model...')
        cPickle.dump(self.model, stream)
        logging.info('Model successfully serializeed')

    def unserialize(self, stream):
        logging.info('Loading model...')
        self.model = cPickle.load(stream)
        logging.info('Model successfully unserializeed')

from random import choice
from numpy import geomspace, linspace


def xgbclassifier_space():
    space = {
        'learning_rate': choice(geomspace(1e-2, 1)),
        'max_depth': choice(range(1, 10)),
        'gamma': choice(geomspace(1e-2, 1)),
        'min_child_weight': choice(range(1, 10)),
        'n_estimators': choice(range(30, 300)),
        'reg_alpha': choice(linspace(0.2, 1)),
        'reg_lambda': choice(linspace(0.2, 2)),
        'scale_pos_weight': choice(linspace(0.3, 2)),
        'n_jobs': 8,
    }
    return space

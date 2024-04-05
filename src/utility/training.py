from tools.preprocessor import Preprocessor
from typing import Tuple
import numpy as np

def scale_data(X_train, X_test, cat_features, num_features) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean')
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train)
    X_test = transformer.transform(X_test)
    return (X_train, X_test)
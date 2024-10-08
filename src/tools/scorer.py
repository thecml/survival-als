import argparse
import numpy as np
import sklearn.base
import torch
import torchtuples as tt
import pandas as pd
import os

from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from lifelines.fitters.weibull_aft_fitter import WeibullAFTFitter
from pycox.models import DeepHitSingle, CoxTime

from sota.mensa.model import MENSA
from utility.config import dotdict
from utility.data import pad_tensor
from utility.conformal import make_mono_quantiles, survival_to_quantile, RegressionErrFunc
from SurvivalEVAL.Evaluations.util import check_monotonicity

class SurvivalNC(sklearn.base.BaseEstimator):
    """Nonconformity scorer using an underlying survival model."""
    def __init__(
            self,
            model,
            error_function=RegressionErrFunc(),
            config: dotdict = {},
            device="cpu"
    ):
        super(SurvivalNC, self).__init__()
        self.model = model
        self.err_func = error_function
        self.config = config
        
        self.interpolate = "Pchip"
        self.device = device
        
        self.mono_method = "bootstrap"
        self.seed = 0

    def fit(self, train_dict: dict, valid_dict: dict):
        """Fits the underlying model of the nonconformity scorer."""
        lr = self.config['lr']
        n_epochs = self.config['n_epochs']
        batch_size = self.config['batch_size']
        self.model.fit(train_dict, valid_dict, learning_rate=lr, n_epochs=n_epochs,
                       patience=10, batch_size=batch_size, verbose=True)

    def score(
            self,
            feature_df: pd.DataFrame,
            t: np.ndarray,
            e: np.ndarray,
            risk: int,
            time_bins: np.ndarray,
            quantile_levels: np.ndarray,
            method: str):
        """Calculates the nonconformity score of a set of samples.

        Parameters
        ----------
        feature_df: pandas DataFrame of shape [n_samples, n_features]
            Inputs of examples for which to calculate a nonconformity score.
        t: numpy array of shape [n_samples]
            Times of examples.
        e: numpy array of shape [n_samples]
            Event indicators of examples.
        quantile_levels: numpy array of shape [n_significance_levels]
            Significance levels (maximum allowed error rate) of predictions.
        method: str
            Decensoring method to use. See `compute_decensor_times` in `utils/util_survival.py` for details.
        Returns
        -------
        conformal_scores : numpy array of shape [n_samples]
            conformity scores of samples.
        """
        x = feature_df.values
        x_names = feature_df.columns.tolist()
        y = np.stack([t, e], axis=1)

        quantile_predictions = self.predict_nc(x=x,
                                               risk=risk,
                                               time_bins=time_bins,
                                               quantile_levels=quantile_levels,
                                               feature_names=x_names)

        if method == 'sampling':
            quantile_predictions = np.repeat(quantile_predictions, 1000, axis=0)

        assert quantile_predictions.shape[0] == y.shape[0], "Sample size does not match."

        conformal_scores = self.err_func.apply(quantile_predictions, y)
        return conformal_scores

    def predict(
            self,
            x: np.ndarray,
            risk: int,
            time_bins: np.ndarray,
            conformal_scores: np.ndarray,
            feature_names: list[str] = None,
            quantile_levels=None
    ):
        quantile_predictions = self.predict_nc(x, risk=risk, time_bins=time_bins,
                                               quantile_levels=quantile_levels,
                                               feature_names=feature_names)

        error_dist = self.err_func.apply_inverse(conformal_scores, quantile_levels)

        quantile_predictions = quantile_predictions - error_dist
        quantile_levels, quantile_predictions = make_mono_quantiles(quantile_levels, quantile_predictions,
                                                                    method=self.mono_method, seed=self.seed)
        # sanity checks
        assert np.all(quantile_predictions >= 0), "Quantile predictions contain negative."
        assert check_monotonicity(quantile_predictions), "Quantile predictions are not monotonic."

        return quantile_predictions

    def predict_nc(
            self,
            x: np.ndarray,
            risk: int,
            time_bins: np.array,
            quantile_levels: np.ndarray,
            feature_names: list[str] = None
    ) -> (np.ndarray, np.ndarray):
        """
        Predict the nonconformity survival curves for a given feature matrix x

        :param x: numpy array of shape [n_samples, n_features]
            feature matrix
        :param feature_names: list of strings
            feature names. Only used for lifelines models.
        :param quantile_levels: numpy array of shape [n_quantiles]
            quantile levels to predict
        :return:
        Non-conformalized predictions for the survival curves
            surv_prob: numpy array of shape [n_samples, n_timepoints]
                survival probability for each sample at each time point
            time_coordinates: numpy array of shape [n_samples, n_timepoints]
                time coordinates for each sample
        """
        # using batch prediction to avoid memory overflow, choose the largest batch size
        # that does not cause memory overflow, this shouldn't impact the performance
        batch_size = 1000
        num_batches = x.shape[0] // batch_size + (x.shape[0] % batch_size > 0)
        quantile_batchs = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, x.shape[0])
            x_batch = x[start_idx:end_idx]
            x_batch = torch.from_numpy(x_batch).type(torch.float64).to(self.device)
            
            surv_prob = self.model.predict(x_batch, time_bins, risk)
            
            time_coordinates = time_bins
            time_coordinates = time_coordinates.cpu().numpy()

            # add 0 to time_coordinates and 1 to surv_prob if not present
            if time_coordinates[0] != 0:
                time_coordinates = np.concatenate([np.array([0]), time_coordinates], 0)
                surv_prob = np.concatenate([np.ones([surv_prob.shape[0], 1]), surv_prob], 1)

            time_coordinates = np.repeat(time_coordinates[np.newaxis, :], surv_prob.shape[0], axis=0)
            quantile_batch = survival_to_quantile(surv_prob, time_coordinates, quantile_levels,
                                                  self.interpolate)
            quantile_batchs.append(quantile_batch)
        # quantile_predictions = np.concatenate(quantile_batchs, 0)
        quantile_predictions = np.vstack(quantile_batchs)

        return quantile_predictions
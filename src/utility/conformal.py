from __future__ import division

import abc
from typing import Optional
import numpy as np

from scipy.interpolate import interp1d, PchipInterpolator
from SurvivalEVAL.Evaluations.util import check_monotonicity

from scipy import interpolate

class RegressionErrFunc(object):
    """Base class for regression model error functions.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(RegressionErrFunc, self).__init__()

    @abc.abstractmethod
    def apply(self, prediction, y):  # , norm=None, beta=0):
        """Apply the nonconformity function.

        Parameters
        ----------
        prediction : numpy array of shape [n_samples, n_classes]
            Class probability estimates for each sample.

        y : numpy array of shape [n_samples]
            True output labels of each sample.

        Returns
        -------
        nc : numpy array of shape [n_samples]
            Nonconformity scores of the samples.
        """
        pass

    @abc.abstractmethod
    def apply_inverse(self, nc, significance):  # , norm=None, beta=0):
        """Apply the inverse of the nonconformity function (i.e.,
        calculate prediction interval).

        Parameters
        ----------
        nc : numpy array of shape [n_calibration_samples]
            Nonconformity scores obtained for conformal predictor.

        significance : float
            Significance level (0, 1).

        Returns
        -------
        interval : numpy array of shape [n_samples, 2]
            Minimum and maximum interval boundaries for each prediction.
        """
        pass


class AbsErrorErrFunc(RegressionErrFunc):
    """Calculates absolute error nonconformity for regression problems.

        For each correct output in ``y``, nonconformity is defined as

        .. math::
            | y_i - \hat{y}_i |
    """

    def __init__(self):
        super(AbsErrorErrFunc, self).__init__()

    def apply(self, prediction, y):
        return np.abs(prediction - y)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc)[::-1]
        border = int(np.floor(significance * (nc.size + 1))) - 1
        border = min(max(border, 0), nc.size - 1)
        return np.vstack([nc[border], nc[border]])


class QuantileRegErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression error.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        max{\hat{q}_low - y, y - \hat{q}_high}

    """

    def __init__(self):
        super(QuantileRegErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - y
        error_high = y - y_upper
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])


# CQR asymmetric error function
class QuantileRegAsymmetricErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression asymmetric error function.

    For each correct output in ``y``, nonconformity is defined as

    .. math::
        E_low = \hat{q}_low - y
        E_high = y - \hat{q}_high

    """

    def __init__(self):
        super(QuantileRegAsymmetricErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]

        error_high = y - y_upper
        error_low = y_lower - y

        err_high = np.reshape(error_high, (y_upper.shape[0], 1))
        err_low = np.reshape(error_low, (y_lower.shape[0], 1))

        return np.concatenate((err_low, err_high), 1)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance / 2) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index, 0], nc[index, 1]])


class OnsSideQuantileRegErrFunc(RegressionErrFunc):
    def __init__(self):
        super(OnsSideQuantileRegErrFunc, self).__init__()

    def apply(self, predictions, y):
        """

        :param predictions:
        :param y:
        :return: (n_samples, n_significance)
        """
        t_ = np.expand_dims(y[:, 0], axis=-1)
        errors = predictions - t_
        return errors

    def apply_inverse(self, nc: np.ndarray, quantile_levels: np.ndarray):
        """

        :param nc: numpy array of shape [n_calibration_samples, n_significance]
        :param quantile_levels:
        :return:
        """
        nc = np.sort(nc, axis=0)
        # -1 because python is 0-indexed
        index1 = np.ceil((1 - quantile_levels) * (nc.shape[0] + 1)) - 1
        index1 = index1.astype(int)
        index1 = index1.clip(0, nc.shape[0] - 1)
        # because i-th number in index1 is the index for i-th significance
        index2 = np.arange(quantile_levels.shape[0])
        errors = nc[index1, index2]
        # using quantile directly
        # quantiles = np.ceil((1 - quantile_levels) * (nc.shape[0] + 1)) / nc.shape[0]
        # errors = np.quantile(nc, quantiles, axis=0).diagonal()
        return errors

def survival_to_quantile(surv_prob, time_coordinates, quantile_levels, interpolate='Pchip'):
    if interpolate == 'Linear':
        Interpolator = interp1d
    elif interpolate == 'Pchip':
        Interpolator = PchipInterpolator
    else:
        raise ValueError(f"Unknown interpolation method: {interpolate}")

    cdf = 1 - surv_prob
    slope = cdf[:, -1] / time_coordinates[:, -1]
    assert cdf.shape == time_coordinates.shape, "CDF and time coordinates have different shapes."
    quantile_predictions = np.empty((cdf.shape[0], quantile_levels.shape[0]))
    for i in range(cdf.shape[0]):
        # fit a scipy interpolation function to the cdf
        cdf_i = cdf[i, :]
        time_coordinates_i = time_coordinates[i, :]
        # remove duplicates in cdf_i (x-axis), otherwise Interpolator will raise an error
        # here we only keep the first occurrence of each unique value
        cdf_i, idx = np.unique(cdf_i, return_index=True)
        time_coordinates_i = time_coordinates_i[idx]
        interp = Interpolator(cdf_i, time_coordinates_i)

        # if the quantile level is beyond last cdf, we extrapolate the
        beyond_last_idx = np.where(quantile_levels > cdf_i[-1])[0]
        quantile_predictions[i] = interp(quantile_levels)
        quantile_predictions[i, beyond_last_idx] = quantile_levels[beyond_last_idx] / slope[i]

    # sanity checks
    assert np.all(quantile_predictions >= 0), "Quantile predictions contain negative."
    assert check_monotonicity(quantile_predictions), "Quantile predictions are not monotonic."
    return quantile_predictions

def make_mono_quantiles(
        quantiles: np.ndarray,
        quan_preds: np.ndarray,
        method: Optional[str] = "ceil",
        seed: Optional[int] = None,
        num_bs: Optional[int] = None
) -> (np.ndarray, np.ndarray):
    """
    Make quantile predictions monotonic and non-negative.
    :param quantiles: np.ndarray of shape (num_quantiles, )
        quantiles to be evaluated
    :param quan_preds: np.ndarray of shape (num_samples, num_quantiles)
        quantile predictions
    :param method: str, optional
        method to make quantile predictions monotonic
    :param seed: int, optional
        random seed
    :param num_bs: int, optional
        number of bootstrap samples to use
    :return:
        quantiles: np.ndarray of shape (num_quantiles, )
            quantiles to be evaluated
        quan_preds: np.ndarray of shape (num_samples, num_quantiles)
            quantile predictions
    """
    # check if quantiles are monotonically increasing
    if np.any(np.sort(quantiles) != quantiles):
        raise ValueError("Defined quantiles must be monotonically increasing.")

    if num_bs is None:
        num_bs = 1000000

    if seed is not None:
        np.random.seed(seed)

    # make sure predictions are non-negative
    quan_preds = np.clip(quan_preds, a_min=0, a_max=None)

    if 0 not in quantiles:
        quantiles = np.insert(quantiles, 0, 0, axis=0)
        quan_preds = np.insert(quan_preds, 0, 0, axis=1)

    if method == "ceil":
        quan_preds = np.maximum.accumulate(quan_preds, axis=1)
    elif method == "floor":
        quan_preds = np.minimum.accumulate(quan_preds[:, ::-1], axis=1)[:, ::-1]
    elif method == "bootstrap":
        # method 1: take too much memory, might cause memory explosion for large dataset
        # need_rearrange = np.any((np.sort(quan_preds, axis=1) != quan_preds), axis=1)
        #
        # extention_at_1 = quan_preds[need_rearrange, -1] / quantiles[-1]
        # inter_lin = interpolate.interp1d(np.r_[quantiles, 1], np.c_[quan_preds[need_rearrange, :], extention_at_1],
        #                                  kind='linear')
        # bootstrap_qf = inter_lin(np.random.uniform(0, 1, num_bs))
        # quan_preds[need_rearrange, :] = np.percentile(bootstrap_qf, 100 * quantiles, axis=1).T
        #
        # method 2: take too much time
        need_rearrange = np.where(np.any((np.sort(quan_preds, axis=1) != quan_preds), axis=1))[0]
        extention_at_1 = quan_preds[:, -1] / quantiles[-1]
        boostrap_samples = np.random.uniform(0, 1, num_bs)
        for idx in need_rearrange:
            inter_lin = interpolate.interp1d(np.r_[quantiles, 1], np.r_[quan_preds[idx, :], extention_at_1[idx]],
                                             kind='linear')
            bootstrap_qf = inter_lin(boostrap_samples)
            quan_preds[idx, :] = np.percentile(bootstrap_qf, 100 * quantiles)
        #
        # method 3: balance between time and memory, but you have to find the right batch size
        # need_rearrange = np.where(np.any((np.sort(quan_preds, axis=1) != quan_preds), axis=1))[0]
        # batch_size = 1024
        # num_batch = need_rearrange.shape[0] // batch_size + (need_rearrange.shape[0] % batch_size > 0)
        # extention_at_1 = quan_preds[:, -1] / quantiles[-1]
        # boostrap_samples = np.random.uniform(0, 1, num_bs)
        # for i in range(num_batch):
        #     idx = need_rearrange[i * batch_size: (i + 1) * batch_size]
        #     inter_lin = interpolate.interp1d(np.r_[quantiles, 1], np.c_[quan_preds[idx, :], extention_at_1[idx]],
        #                                      kind='linear')
        #     bootstrap_qf = inter_lin(boostrap_samples)
        #     quan_preds[idx, :] = np.percentile(bootstrap_qf, 100 * quantiles, axis=1).T
    else:
        raise ValueError(f"Unknown method {method}.")

    # fix some numerical issues
    # In some cases, the quantile predictions
    small_values = np.arange(0, quantiles.size) * 1e-10
    quan_preds = quan_preds + small_values

    return quantiles, quan_preds


import numpy as np
import pandas as pd
import math
import torch
from typing import List, Tuple, Optional, Union
from sklearn.utils import shuffle
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split
from tools.preprocessor import Preprocessor
from SurvivalEVAL.Evaluations.util import KaplanMeierArea, km_mean

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

Numeric = Union[float, int, bool]
NumericArrayLike = Union[List[Numeric], Tuple[Numeric], np.ndarray, pd.Series, pd.DataFrame, torch.Tensor]

def cox_survival(
        baseline_survival: torch.Tensor,
        linear_predictor: torch.Tensor,
        dtype: torch.dtype
) -> torch.Tensor:
    """
    Calculate the individual survival distributions based on the baseline survival curves and the liner prediction values.
    :param baseline_survival: (n_time_bins, )
    :param linear_predictor: (n_samples, n_data)
    :return:
    The invidual survival distributions. shape = (n_samples, n_time_bins)
    """
    n_sample = linear_predictor.shape[0]
    n_data = linear_predictor.shape[1]
    risk_score = torch.exp(linear_predictor)
    survival_curves = torch.empty((n_sample, n_data, baseline_survival.shape[0]), dtype=dtype).to(linear_predictor.device)
    for i in range(n_sample):
        for j in range(n_data):
            survival_curves[i, j, :] = torch.pow(baseline_survival, risk_score[i, j])
    return survival_curves

def cox_nll(
        risk_pred: torch.Tensor,
        precision: torch.Tensor,
        log_var: torch.Tensor,
        true_times: torch.Tensor,
        true_indicator: torch.Tensor,
        model: torch.nn.Module,
        C1: float
) -> torch.Tensor:
    """Computes the negative log-likelihood of a batch of model predictions.

    Parameters
    ----------
    risk_pred : torch.Tensor, shape (num_samples, )
        Risk prediction from Cox-based model. It means the relative hazard ratio: \beta * x.
    true_times : torch.Tensor, shape (num_samples, )
        Tensor with the censor/event time.
    true_indicator : torch.Tensor, shape (num_samples, )
        Tensor with the censor indicator.
    model
        PyTorch Module with at least `MTLR` layer.
    C1
        The L2 regularization strength.

    Returns
    -------
    torch.Tensor
        The negative log likelihood.
    """
    eps = 1e-20
    risk_pred = risk_pred.reshape(-1, 1)
    true_times = true_times.reshape(-1, 1)
    true_indicator = true_indicator.reshape(-1, 1)
    mask = torch.ones(true_times.shape[0], true_times.shape[0]).to(true_times.device)
    mask[(true_times.T - true_times) > 0] = 0
    max_risk = risk_pred.max()
    log_loss = torch.exp(risk_pred - max_risk) * mask
    log_loss = torch.sum(log_loss, dim=0)
    log_loss = torch.log(log_loss + eps).reshape(-1, 1) + max_risk
    # Sometimes in the batch we got all censoring data, so the denominator gets 0 and throw nan.
    # Solution: Consider increase the batch size. Afterall the nll should performed on the whole dataset.
    # Based on equation 2&3 in https://arxiv.org/pdf/1606.00931.pdf
    neg_log_loss = -torch.sum(precision * (risk_pred - log_loss) * true_indicator) / torch.sum(true_indicator) + log_var

    # L2 regularization
    for k, v in model.named_parameters():
        if "weight" in k:
            neg_log_loss += C1/2 * torch.norm(v, p=2)

    return neg_log_loss

def calculate_baseline_hazard(
        logits: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor
) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    """
    Calculate the baseline cumulative hazard function and baseline survival function using Breslow estimator
    :param logits: logit outputs calculated from the Cox-based network using training data.
    :param time: Survival time of training data.
    :param event: Survival indicator of training data.
    :return:
    uniq_times: time bins correspond of the baseline hazard/survival.
    cum_baseline_hazard: cumulative baseline hazard
    baseline_survival: baseline survival curve.
    """
    risk_score = torch.exp(logits)
    order = torch.argsort(time)
    risk_score = risk_score[order]
    uniq_times, n_events, n_at_risk, _ = compute_unique_counts(event, time, order)

    divisor = torch.empty(n_at_risk.shape, dtype=torch.float, device=n_at_risk.device)
    value = torch.sum(risk_score)
    divisor[0] = value
    k = 0
    for i in range(1, len(n_at_risk)):
        d = n_at_risk[i - 1] - n_at_risk[i]
        value -= risk_score[k:(k + d)].sum()
        k += d
        divisor[i] = value

    assert k == n_at_risk[0] - n_at_risk[-1]

    hazard = n_events / divisor
    # Make sure the survival curve always starts at 1
    if 0 not in uniq_times:
        uniq_times = torch.cat([torch.tensor([0]).to(uniq_times.device), uniq_times], 0)
        hazard = torch.cat([torch.tensor([0]).to(hazard.device), hazard], 0)
    # TODO: torch.cumsum with cuda array will generate a non-monotonic array. Need to update when torch fix this bug
    # See issue: https://github.com/pytorch/pytorch/issues/21780
    baseline_hazard = hazard.cpu()
    cum_baseline_hazard = torch.cumsum(hazard.cpu(), dim=0).to(hazard.device)
    baseline_survival = torch.exp(- cum_baseline_hazard)
    if baseline_survival.isinf().any():
        print(f"Baseline survival contains \'inf\', need attention. \n"
              f"Baseline survival distribution: {baseline_survival}")
        last_zero = torch.where(baseline_survival == 0)[0][-1].item()
        baseline_survival[last_zero + 1:] = 0
    baseline_survival = make_monotonic(baseline_survival)
    return uniq_times, cum_baseline_hazard, baseline_survival

'''
Impute missing values and scale
'''
def preprocess_data(X_train, X_valid, X_test, cat_features,
                    num_features, as_array=False) \
    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean', scaling_strategy="standard")
    transformer = preprocessor.fit(X_train, cat_feats=cat_features, num_feats=num_features,
                                   one_hot=True, fill_value=-1)
    X_train = transformer.transform(X_train)
    X_valid = transformer.transform(X_valid)
    X_test = transformer.transform(X_test)
    if as_array:
        return (np.array(X_train), np.array(X_valid), np.array(X_test))
    return (X_train, X_valid, X_test)

def split_time_event(y):
    y_t = np.array(y['TTE'])
    y_e = np.array(y['Event'])
    return (y_t, y_e)

def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None
) -> torch.Tensor:
    """Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=torch.float)
    return bins

def make_monotonic(
        array: Union[torch.Tensor, np.ndarray, list]
):
    for i in range(len(array) - 1):
        if not array[i] >= array[i + 1]:
            array[i + 1] = array[i]
    return array

def multilabel_train_test_split(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    """
    X, y = shuffle(X, y, random_state=random_state)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test

def make_multi_event_stratified_column(times):
    N, d = times.shape[0], times.shape[1]
    num_elements_per_column = N // d
    remaining_elements = N % d
    result = []
    for i in range(d):
        if i < remaining_elements:
            result.extend(times[:num_elements_per_column + 1, i])
        else:
            result.extend(times[:num_elements_per_column, i])
    result_array = np.array(result)
    return result_array

def make_stratified_split(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_valid: float = 0.0,
        frac_test: float = 0.5,
        n_events: int = 0,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    assert frac_train >= 0 and frac_valid >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_valid + frac_test
    frac_train = frac_train / frac_sum
    frac_valid = frac_valid / frac_sum
    frac_test = frac_test / frac_sum

    X = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True).reshape(-1, 1)
    elif stratify_colname == "both":
        t = df["time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["event"]
        stra_lab = np.stack([t, e], axis=1)
    elif stratify_colname == "multi":
        stra_lab = []
        for i in range(n_events):
            t = df[f"t{i+1}"]
            e = df[f"e{i+1}"]
            bins = np.linspace(start=t.min(), stop=t.max(), num=20)
            t = np.digitize(t, bins, right=True)
            stra_lab.append(e)
            stra_lab.append(t)
        stra_lab = np.stack(stra_lab, axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train, _, x_temp, y_temp = multilabel_train_test_split(X, y=stra_lab, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_valid == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_valid + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test

def make_stratification_label(df):
    t = df["Survival_time"]
    bins = np.linspace(start=t.min(), stop=t.max(), num=20)
    t = np.digitize(t, bins, right=True)
    e = df["Event"]
    stra_lab = np.stack([t, e], axis=1)
    return stra_lab

def encode_survival(
        time: Union[float, int, NumericArrayLike],
        event: Union[int, bool, NumericArrayLike],
        bins: NumericArrayLike
) -> torch.Tensor:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    # TODO this should handle arrays and (CUDA) tensors
    if isinstance(time, (float, int, np.ndarray)):
        time = np.atleast_1d(time)
        time = torch.tensor(time)
    if isinstance(event, (int, bool, np.ndarray)):
        event = np.atleast_1d(event)
        event = torch.tensor(event)

    if isinstance(bins, np.ndarray):
        bins = torch.tensor(bins)

    try:
        device = bins.device
    except AttributeError:
        device = "cpu"

    time = np.clip(time, 0, bins.max())
    # add extra bin [max_time, inf) at the end
    y = torch.zeros((time.shape[0], bins.shape[0] + 1),
                    dtype=torch.float,
                    device=device)
    # For some reason, the `right` arg in torch.bucketize
    # works in the _opposite_ way as it does in numpy,
    # so we need to set it to True
    bin_idxs = torch.bucketize(time, bins, right=True)
    for i, (bin_idx, e) in enumerate(zip(bin_idxs, event)):
        if e == 1:
            y[i, bin_idx] = 1
        else:
            y[i, bin_idx:] = 1
    return y.squeeze()

def reformat_survival(
        dataset: pd.DataFrame,
        time_bins: NumericArrayLike,
        dtype: torch.dtype
) -> (torch.Tensor, torch.Tensor):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    x = torch.tensor(dataset.drop(["time", "event"], axis=1).values, dtype=dtype)
    y = encode_survival(dataset["time"].values, dataset["event"].values, time_bins)
    return x, y

def coverage(time_bins, upper, lower, true_times, true_indicator) -> float:
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    time_bins = check_and_convert(time_bins)
    upper, lower = check_and_convert(upper, lower)
    true_times, true_indicator = check_and_convert(true_times, true_indicator)
    true_indicator = true_indicator.astype(bool)
    covered = 0
    upper_median_times = predict_median_survival_times(upper, time_bins, round_up=True)
    lower_median_times = predict_median_survival_times(lower, time_bins, round_up=False)
    covered += 2 * np.logical_and(upper_median_times[true_indicator] >= true_times[true_indicator],
                                  lower_median_times[true_indicator] <= true_times[true_indicator]).sum()
    covered += np.sum(upper_median_times[~true_indicator] >= true_times[~true_indicator])
    total = 2 * true_indicator.sum() + (~true_indicator).sum()
    return covered / total

def predict_median_survival_times(
        survival_curves: np.ndarray,
        times_coordinate: np.ndarray,
        round_up: bool = True
):
    median_probability_times = np.zeros(survival_curves.shape[0])
    max_time = times_coordinate[-1]
    slopes = (1 - survival_curves[:, -1]) / (0 - max_time)

    if round_up:
        # Find the first index in each row that are smaller or equal than 0.5
        times_indices = np.where(survival_curves <= 0.5, survival_curves, -np.inf).argmax(axis=1)
    else:
        # Find the last index in each row that are larger or equal than 0.5
        times_indices = np.where(survival_curves >= 0.5, survival_curves, np.inf).argmin(axis=1)

    need_extend = survival_curves[:, -1] > 0.5
    median_probability_times[~need_extend] = times_coordinate[times_indices][~need_extend]
    median_probability_times[need_extend] = (max_time + (0.5 - survival_curves[:, -1]) / slopes)[need_extend]

    return median_probability_times

def convert_to_structured (T, E):
    default_dtypes = {"names": ("event", "time"), "formats": ("bool", "f8")}
    concat = list(zip(E, T))
    return np.array(concat, dtype=default_dtypes)

def make_event_times (t_train, e_train):
    unique_times = compute_unique_counts(torch.Tensor(e_train), torch.Tensor(t_train))[0]
    if 0 not in unique_times:
        unique_times = torch.cat([torch.tensor([0]).to(unique_times.device), unique_times], 0)
    return unique_times.numpy()

def make_time_bins(
        times: NumericArrayLike,
        num_bins: Optional[int] = None,
        use_quantiles: bool = True,
        event: Optional[NumericArrayLike] = None,
        dtype=torch.float64
) -> torch.Tensor:
    """
    Courtesy of https://ieeexplore.ieee.org/document/10158019
    
    Creates the bins for survival time discretisation.

    By default, sqrt(num_observation) bins corresponding to the quantiles of
    the survival time distribution are used, as in https://github.com/haiderstats/MTLR.

    Parameters
    ----------
    times
        Array or tensor of survival times.
    num_bins
        The number of bins to use. If None (default), sqrt(num_observations)
        bins will be used.
    use_quantiles
        If True, the bin edges will correspond to quantiles of `times`
        (default). Otherwise, generates equally-spaced bins.
    event
        Array or tensor of event indicators. If specified, only samples where
        event == 1 will be used to determine the time bins.

    Returns
    -------
    torch.Tensor
        Tensor of bin edges.
    """
    # TODO this should handle arrays and (CUDA) tensors
    if event is not None:
        times = times[event == 1]
    if num_bins is None:
        num_bins = math.ceil(math.sqrt(len(times)))
    if use_quantiles:
        # NOTE we should switch to using torch.quantile once it becomes
        # available in the next version
        bins = np.unique(np.quantile(times, np.linspace(0, 1, num_bins)))
    else:
        bins = np.linspace(times.min(), times.max(), num_bins)
    bins = torch.tensor(bins, dtype=dtype)
    return bins

def compute_unique_counts(
        event: torch.Tensor,
        time: torch.Tensor,
        order: Optional[torch.Tensor] = None):
    """Count right censored and uncensored samples at each unique time point.

    Parameters
    ----------
    event : array
        Boolean event indicator.

    time : array
        Survival time or time of censoring.

    order : array or None
        Indices to order time in ascending order.
        If None, order will be computed.

    Returns
    -------
    times : array
        Unique time points.

    n_events : array
        Number of events at each time point.

    n_at_risk : array
        Number of samples that have not been censored or have not had an event at each time point.

    n_censored : array
        Number of censored samples at each time point.
    """
    n_samples = event.shape[0]

    if order is None:
        order = torch.argsort(time)

    uniq_times = torch.empty(n_samples, dtype=time.dtype, device=time.device)
    uniq_events = torch.empty(n_samples, dtype=torch.int, device=time.device)
    uniq_counts = torch.empty(n_samples, dtype=torch.int, device=time.device)

    i = 0
    prev_val = time[order[0]]
    j = 0
    while True:
        count_event = 0
        count = 0
        while i < n_samples and prev_val == time[order[i]]:
            if event[order[i]]:
                count_event += 1

            count += 1
            i += 1

        uniq_times[j] = prev_val
        uniq_events[j] = count_event
        uniq_counts[j] = count
        j += 1

        if i == n_samples:
            break

        prev_val = time[order[i]]

    uniq_times = uniq_times[:j]
    uniq_events = uniq_events[:j]
    uniq_counts = uniq_counts[:j]
    n_censored = uniq_counts - uniq_events

    # offset cumulative sum by one
    total_count = torch.cat([torch.tensor([0], device=uniq_counts.device), uniq_counts], dim=0)
    n_at_risk = n_samples - torch.cumsum(total_count, dim=0)

    return uniq_times, uniq_events, n_at_risk[:-1], n_censored

def check_and_convert(*args):
    """ Makes sure that the given inputs are numpy arrays, list,
        tuple, panda Series, pandas DataFrames, or torch Tensors.

        Also makes sure that the given inputs have the same shape.

        Then convert the inputs to numpy array.

        Parameters
        ----------
        * args : tuple of objects
                 Input object to check / convert.

        Returns
        -------
        * result : tuple of numpy arrays
                   The converted and validated arg.

        If the input isn't numpy arrays, list or pandas DataFrames, it will
        fail and ask to provide the valid format.
    """

    result = ()
    last_length = ()
    for i, arg in enumerate(args):

        if len(arg) == 0:
            error = " The input is empty. "
            error += "Please provide at least 1 element in the array."
            raise IndexError(error)

        else:

            if isinstance(arg, np.ndarray):
                x = (arg.astype(np.double),)
            elif isinstance(arg, list):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, tuple):
                x = (np.asarray(arg).astype(np.double),)
            elif isinstance(arg, pd.Series):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, pd.DataFrame):
                x = (arg.values.astype(np.double),)
            elif isinstance(arg, torch.Tensor):
                x = (arg.cpu().numpy().astype(np.double),)
            else:
                error = """{arg} is not a valid data format. Only use 'list', 'tuple', 'np.ndarray', 'torch.Tensor', 
                        'pd.Series', 'pd.DataFrame'""".format(arg=type(arg))
                raise TypeError(error)

            if np.sum(np.isnan(x)) > 0.:
                error = "The #{} argument contains null values"
                error = error.format(i + 1)
                raise ValueError(error)

            if len(args) > 1:
                if i > 0:
                    assert x[0].shape == last_length, """Shapes between {}-th input array and 
                    {}-th input array are not consistent""".format(i - 1, i)
                result += x
                last_length = x[0].shape
            else:
                result = x[0]

    return result

def compute_decensor_times(test_set, train_set, method="margin"):
    t_train, e_train = train_set["time"].values, train_set["event"].values.astype(bool)
    n_train = len(t_train)
    km_train = KaplanMeierArea(t_train, e_train)

    t_test, e_test = test_set["time"].values, test_set["event"].values.astype(bool)
    n_test = len(t_test)

    if method == "uncensored":
        # drop censored samples directly
        decensor_set = test_set.drop(test_set[~e_test].index)
        decensor_set.reset_index(drop=True, inplace=True)
        feature_df = decensor_set.drop(["time", "event"], axis=1)
        t = decensor_set["time"].values
        e = decensor_set["event"].values
    elif method == "margin":
        feature_df = test_set.drop(["time", "event"], axis=1)
        best_guesses = t_test.copy().astype(float)
        km_linear_zero = -1 / ((1 - min(km_train.survival_probabilities)) / (0 - max(km_train.survival_times)))
        if np.isinf(km_linear_zero):
            km_linear_zero = max(km_train.survival_times)

        censor_test = t_test[~e_test]
        conditional_mean_t = km_train.best_guess(censor_test)
        conditional_mean_t[censor_test > km_linear_zero] = censor_test[censor_test > km_linear_zero]

        best_guesses[~e_test] = conditional_mean_t
        t = best_guesses
        e = np.ones_like(best_guesses)
    elif method == "PO":
        feature_df = test_set.drop(["time", "event"], axis=1)
        best_guesses = t_test.copy().astype(float)
        events, population_counts = km_train.events.copy(), km_train.population_count.copy()
        times = km_train.survival_times.copy()
        probs = km_train.survival_probabilities.copy()
        # get the discrete time points where the event happens, then calculate the area under those discrete time only
        # this doesn't make any difference for step function, but it does for trapezoid rule.
        unique_idx = np.where(events != 0)[0]
        if unique_idx[-1] != len(events) - 1:
            unique_idx = np.append(unique_idx, len(events) - 1)
        times = times[unique_idx]
        population_counts = population_counts[unique_idx]
        events = events[unique_idx]
        probs = probs[unique_idx]
        sub_expect_time = km_mean(times.copy(), probs.copy())

        # use the idea of dynamic programming to calculate the multiplier of the KM estimator in advances.
        # if we add a new time point to the KM curve, the multiplier before the new time point will be
        # 1 - event_counts / (population_counts + 1), and the multiplier after the new time point will be
        # the same as before.
        multiplier = 1 - events / population_counts
        multiplier_total = 1 - events / (population_counts + 1)

        for i in range(n_test):
            if e_test[i] != 1:
                total_multiplier = multiplier.copy()
                insert_index = np.searchsorted(times, t_test[i], side='right')
                total_multiplier[:insert_index] = multiplier_total[:insert_index]
                survival_probabilities = np.cumprod(total_multiplier)
                if insert_index == len(times):
                    times_addition = np.append(times, t_test[i])
                    survival_probabilities_addition = np.append(survival_probabilities, survival_probabilities[-1])
                    total_expect_time = km_mean(times_addition, survival_probabilities_addition)
                else:
                    total_expect_time = km_mean(times, survival_probabilities)
                best_guesses[i] = (n_train + 1) * total_expect_time - n_train * sub_expect_time

        t = best_guesses
        e = np.ones_like(best_guesses)
    elif method == "sampling":
        # repeat each sample 1000 times,
        # for event subject, the event time will be the same for 1000 times;
        # for censored subject, the "fake" event time will be sampled from the conditional KM curve,
        # the conditional KM curve is the KM distribution (km_train) given we know the subject is censored at time c
        # and make the censor bit to 1
        feature_df = test_set.drop(["time", "event"], axis=1)
        t = np.repeat(test_set["time"].values, 1000)
        uniq_times = km_train.survival_times
        surv = km_train.survival_probabilities
        last_time = km_train.km_linear_zero
        if uniq_times[0] != 0:
            uniq_times = np.insert(uniq_times, 0, 0, axis=0)
            surv = np.insert(surv, 0, 1, axis=0)

        for i in range(n_test):
            # x_i = x_test[i, :]
            if e_test[i] != 1:
                s_prob = km_train.predict(t_test[i])
                cond_surv = surv / s_prob
                cond_surv = np.clip(cond_surv, 0, 1)
                cond_cdf = 1 - cond_surv
                cond_pdf = np.diff(np.append(cond_cdf, 1))

                # sample from the conditional KM curve
                surrogate_t = np.random.choice(uniq_times, size=1000, p=cond_pdf)

                if last_time != uniq_times[-1]:
                    need_extension = surrogate_t == uniq_times[-1]
                    surrogate_t[need_extension] = np.random.uniform(uniq_times[-1], last_time, need_extension.sum())

                t[i * 1000:(i + 1) * 1000] = surrogate_t

        e = np.ones_like(t)

    else:
        raise ValueError(f"Unknown method {method}.")

    return feature_df, t, e
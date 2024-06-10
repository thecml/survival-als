import pandas as pd
import numpy as np
import config as cfg
from tools.event_loader import DataLoader
from utility.survival import split_time_event
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_probability as tfp
from utility.mcmc import sample_hmc
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sksurv.metrics import concordance_index_censored
from utility.survival import split_time_event, make_stratified_split, convert_to_structured, calculate_event_times
import seaborn as sns
from utility.plot import _TFColor
from matplotlib.lines import Line2D
from pathlib import Path
from utility.training import scale_data

TFColor = _TFColor()

tfd = tfp.distributions
tfb = tfp.bijectors

DTYPE = tf.float32
N_CHAINS = 1

if __name__ == "__main__":
    # Load data
    dl = DataLoader().load_data(event="Walking")
    num_features, cat_features = dl.get_features()
    df = dl.get_data()
    
    # Split data in train/test sets
    df_train, _, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.8,
                                                 frac_valid=0, frac_test=0.2, random_state=0)
    X_train = df_train[cat_features+num_features]
    X_test = df_test[cat_features+num_features]
    y_train = convert_to_structured(df_train["time"], df_train["event"])
    y_test = convert_to_structured(df_test["time"], df_test["event"])

    # Scale data
    X_train, X_test = scale_data(X_train, X_test, cat_features, num_features)

    # Split data in train/test sets
    t_train, e_train = split_time_event(y_train) 
    t_test, e_test = split_time_event(y_test)
    
    # Make event times
    event_times = calculate_event_times(t_train, e_train)
    
    # Split training data in observed/unobserved
    y_obs = tf.convert_to_tensor(t_train[e_train], dtype=DTYPE)
    y_cens = tf.convert_to_tensor(t_train[~e_train], dtype=DTYPE)
    x_obs = tf.convert_to_tensor(X_train[e_train], dtype=DTYPE)
    x_cens = tf.convert_to_tensor(X_train[~e_train], dtype=DTYPE)
    
    n_dims = x_cens.shape[1]

    obs_model = tfd.JointDistributionSequentialAutoBatched([
            tfd.Normal(loc=tf.zeros([1]), scale=tf.ones([1])), # alpha
            tfd.Normal(loc=tf.zeros([n_dims,1]), scale=tf.ones([n_dims,1])), # beta
            lambda beta, alpha: tfd.Exponential(rate=1/tf.math.exp(tf.transpose(x_obs)*beta + alpha))]
        )

    def log_prob(x_obs, x_cens, y_obs, y_cens, alpha, beta):
        lp = obs_model.log_prob([alpha, beta, y_obs])
        potential = exponential_lccdf(x_cens, y_cens, alpha, beta)
        return lp + potential

    def exponential_lccdf(x_cens, y_cens, alpha, beta):
        return tf.reduce_sum(-y_cens / tf.exp(tf.transpose(x_cens)*beta + alpha))

    number_of_steps = 10000
    number_burnin_steps = int(number_of_steps/10)

    # Sample from the prior
    initial_coeffs = obs_model.sample(1)

    # Run sampling for number of chains
    unnormalized_post_log_prob = lambda *args: log_prob(x_obs, x_cens, y_obs, y_cens, *args)
    chains = [sample_hmc(unnormalized_post_log_prob, [tf.zeros_like(initial_coeffs[0]),
                                                      tf.zeros_like(initial_coeffs[1])],
                         n_steps=number_of_steps, n_burnin_steps=number_burnin_steps) for _ in range(N_CHAINS)]

    # Calculate target accept prob
    for chain_id in range(N_CHAINS):
        log_accept_ratio = chains[chain_id][1][1][number_burnin_steps:]
        target_accept_prob = tf.math.exp(tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.))).numpy()
        print(f'Target acceptance probability for {chain_id}: {round(100*target_accept_prob)}%')
    
    # Calculate accepted rate
    plt.figure(figsize=(10,6))
    for chain_id in range(N_CHAINS):
        accepted_samples = chains[chain_id][1][0][number_burnin_steps:]
        print(f'Acceptance rate chain for {chain_id}: {round(100*np.mean(accepted_samples), 2)}%')
        n_accepted_samples = len(accepted_samples)
        n_bins = int(n_accepted_samples/100)
        sample_indicies = np.linspace(0, n_accepted_samples, n_bins)
        means = [np.mean(accepted_samples[:int(idx)]) for idx in sample_indicies[5:]]
        plt.plot(np.arange(len(means)), means)
    plt.savefig(Path.joinpath(cfg.RESULTS_DIR, f"acceptance_rate.png"), format='png', bbox_inches="tight")
    plt.close()
    
    # Take mean of combined chains to get alpha and beta values
    chains = [chain[0] for chain in chains] # leave out traces

    chain_index = 0
    samples_index = 0
    beta_index = 1
    n_dims = chains[chain_index][beta_index].shape[2] # get n dims from first chain
    chains_t = list(map(list, zip(*chains)))
    chains_samples = [tf.squeeze(tf.concat(samples, axis=0)) for samples in chains_t]
    alpha = tf.reduce_mean(chains_samples[0]).numpy().flatten()
    betas_mean = tf.reduce_mean(chains_samples[1], axis=0).numpy().flatten()
    alphas = chains_samples[0].numpy().flatten()[number_burnin_steps:]
    betas = chains_samples[1].numpy().flatten()[number_burnin_steps:]
    
    print(alphas)
    print(betas)
    
    # Make predictions on test set
    predict_func = lambda data: np.exp(alpha + np.dot(betas_mean, np.transpose(data)))
    test_preds = np.zeros((len(X_test)))
    for i, data in enumerate(X_test):
        test_preds[i] = predict_func(data)

    # Calculate concordance index
    c_harrell = concordance_index_censored(e_test, t_test, -test_preds)
    print(c_harrell)
    
    # Calculate log rate as log lambda = a+bX 
    lambda_no_beta = np.exp(alphas)
    lambda_beta0 = np.exp(alphas + betas)
    #lambda_beta1 = np.exp(alphas + betas[1])
    #lambda_beta2 = np.exp(alphas + betas[2])

    # Plot distributions with various betas used. Notice the shift in risk (normalized)
    sns.histplot(lambda_beta0, color='y', stat='density', label='Started in lower extremity')
    sns.histplot(lambda_no_beta, color='r', stat='density', label='Started elsewhere')
    plt.legend(fontsize=12)
    plt.xlabel("Time to event", size=12)
    plt.ylabel("Density", size=12)
    plt.savefig(Path.joinpath(cfg.RESULTS_DIR, f"alpha_beta_dist.png"), format='png', bbox_inches="tight")
    plt.close()
    
    # Plot the posterior samples
    plt.figure(figsize=(10,6))
    plt.subplot(2, 2, 1)
    plt.hist(alphas, bins=100, color=TFColor[6], alpha=0.8)
    plt.ylabel('Frequency')
    plt.title('posterior α samples', fontsize=14)

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(number_of_steps-number_burnin_steps), alphas, color=TFColor[6], alpha=0.8)
    plt.ylabel('Sample Value')
    plt.title('posterior α samples', fontsize=14)
    plt.savefig(Path.joinpath(cfg.RESULTS_DIR, f"alpha_samples.png"), format='png', bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10,6))
    plt.subplot(2, 2, 1)
    plt.hist(betas, bins=100, color=TFColor[3], alpha=0.8)
    plt.ylabel('Frequency')
    plt.title(f'posterior β samples', fontsize=14)

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(number_of_steps-number_burnin_steps), betas, color=TFColor[3], alpha=0.8)
    plt.ylabel('Sample Value')
    plt.title(f'posterior β samples', fontsize=14)
    plt.savefig(Path.joinpath(cfg.RESULTS_DIR, f"beta_samples.png"), format='png', bbox_inches="tight")
    plt.close()
    
    # Plot survival function for lower extremity / non-lower extremity start
    t = np.linspace(100,600)
    lam_nb = np.mean(lambda_no_beta)
    lam_b = np.mean(lambda_beta0)
    plt.plot(t, np.exp(-t/lam_nb), c='r', linewidth=2, alpha=1)
    plt.plot(t, np.exp(-t/lam_b), c='y', linewidth=2, alpha=1)
    plt.ylabel("S(t)")
    plt.xlabel("Days")
    legend_elements = [Line2D([0], [0], color='y', label='Started in lower extremity'),
                       Line2D([0], [0], color='r', label='Started elsewhere')]
    plt.legend(handles=legend_elements, fontsize=12)
    plt.savefig(Path.joinpath(cfg.RESULTS_DIR, f"tte_walking_lower_extremity.png"), format='png', bbox_inches="tight")
    plt.close()
    
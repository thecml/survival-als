from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from pycox.models import DeepHitSingle
import torchtuples as tt

def make_deephit_single(in_features, out_features, time_bins, device, config):
    num_nodes = config['num_nodes_shared']
    batch_norm = config['batch_norm']
    dropout = config['dropout']
    labtrans = DeepHitSingle.label_transform(time_bins)
    net = tt.practical.MLPVanilla(in_features=in_features, num_nodes=num_nodes,
                                  out_features=labtrans.out_features, batch_norm=batch_norm,
                                  dropout=dropout)
    model = DeepHitSingle(net, tt.optim.Adam, device=device, alpha=0.2, sigma=0.1,
                          duration_index=labtrans.cuts)
    model.label_transform = labtrans
    return model

def train_deephit_model(model, x_train, y_train, valid_data, config):
    epochs = config['epochs']
    batch_size = config['batch_size']
    verbose = config['verbose']
    if config['early_stop']:
        callbacks = [tt.callbacks.EarlyStopping(patience=config['patience'])]
    else:
        callbacks = []
    model.fit(x_train, y_train, batch_size, epochs, callbacks, verbose, val_data=valid_data)
    return model

def format_data_deephit_single(train_dict, valid_dict, labtrans, event_id):
    train_dict_dh = dict()
    train_dict_dh['X'] = train_dict['X'].cpu().numpy()
    train_dict_dh['E'] = train_dict['E'][:,event_id].cpu().numpy()
    train_dict_dh['T'] = train_dict['T'][:,event_id].cpu().numpy()
    valid_dict_dh = dict()
    valid_dict_dh['X'] = valid_dict['X'].cpu().numpy()
    valid_dict_dh['E'] = valid_dict['E'][:,event_id].cpu().numpy()
    valid_dict_dh['T'] = valid_dict['T'][:,event_id].cpu().numpy()
    get_target = lambda data: (data['T'], data['E'])
    y_train = labtrans.transform(*get_target(train_dict_dh))
    y_valid = labtrans.transform(*get_target(valid_dict_dh))
    out_features = len(labtrans.cuts)
    duration_index = labtrans.cuts
    train_data = {'X': train_dict_dh['X'], 'T': y_train[0], 'E': y_train[1]}
    valid_data = {'X': valid_dict_dh['X'], 'T': y_valid[0], 'E': y_valid[1]}
    return train_data, valid_data, out_features, duration_index
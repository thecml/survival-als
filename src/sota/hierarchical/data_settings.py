'''
Record of all settings for datasets
Definitions:
    location: where the data is stored
    features: where in location (see above) the covariates/features are stored
    terminal event: event such that no other events can occur after it
    discrete: whether the time values are discrete
    event ranks: each key represents and event, the values are the events that prevent it
    event groups: each key represents the position in a trajectory (e.g., first, second, ...), values represent which events can occur in that position
    min_time: earliest event time
    max_time: latest event time (prediction horizon)
    min_epoch: minimum number of epochs to train for (while learning the model)
'''

als_settings = \
{
    'num_events': 4, \
    'num_bins': 56, \
    'terminal_events': [], \
    'discrete': False, \
    'event_ranks': {0:[], 1:[], 2:[], 3:[]}, \
    'event_groups': {0:[0, 1, 2, 3],
                     1:[0, 1, 2, 3],
                     2:[0, 1, 2, 3],
                     3:[0, 1, 2, 3]}, \
    'min_time': 1, \
    'max_time': 56, \
    'min_epoch': 1, \
    'max_epoch': 5 \
}
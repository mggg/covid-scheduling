import numpy as np
from dateutil.parser import parse as ts_parse
from covid_scheduling.load_balancing import site_weights

EPS = 1e-8


def test_site_weights_uniform(config_simple):
    weights, _ = site_weights(config_simple, ts_parse('2020-08-17'),
                              ts_parse('2020-08-23'))
    assert weights.shape == (7, )
    assert np.all(np.abs(weights - (np.ones(7) / 7).reshape((7, 1))) < EPS)


def test_site_weights_no_weekend(config_simple):
    filtered_hours = [
        h for h in config_simple['sites']['Testing']['hours']
        if h['day'] not in ('Saturday', 'Sunday')
    ]
    config_simple['sites']['Testing']['hours'] = filtered_hours
    weights, _ = site_weights(config_simple, ts_parse('2020-08-17'),
                              ts_parse('2020-08-23'))
    expected_weights = np.ones(7)
    expected_weights[5:] = 0
    assert weights.shape == (7, )
    print(weights)
    assert np.all(np.abs(weights - (expected_weights / 5)) < EPS)


def test_site_weights_two_blocks(config_two_blocks):
    expected_weights = np.ones(14)
    expected_weights[1::2] = 2
    expected_weights /= expected_weights.sum()
    weights, _ = site_weights(config_two_blocks, ts_parse('2020-08-17'),
                              ts_parse('2020-08-23'))
    assert weights.shape == (14, )
    assert np.all(np.abs(weights - expected_weights) < EPS)
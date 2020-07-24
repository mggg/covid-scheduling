""""Common fixtures for unit tests."""
import pytest
from dateutil.parser import parse as ts_parse
from covid_scheduling.constants import DAYS
from covid_scheduling.schemas import validate_people, validate_config


@pytest.fixture
def config_simple_raw():
    """A configuration with one {cohort, schedule block, site}."""
    return {
        'Campus': {
            'policy': {
                'blocks': {
                    # 8 a.m. - 8 p.m.
                    'Block': {
                        'start': '08:00:00',
                        'end': '20:00:00'
                    }
                },
                'cohorts': {
                    'People': {
                        # 1 test/week target
                        'interval': {
                            'min': 1,
                            'target': 3.5,
                            'max': 6
                        }
                    }
                }
            },
            'sites': {
                'Testing': {
                    'n_lines':
                    1,
                    'hours': [
                        # Testing windows overlap perfectly with the
                        # single block in the block schedule.
                        {
                            'day': day,
                            'start': '08:00:00',
                            'end': '20:00:00'
                        } for day in DAYS
                    ]
                }
            }
        }
    }


@pytest.fixture
def config_simple(config_simple_raw):
    """A validated configuration with one {cohort, schedule block, site}."""
    return validate_config(config_simple_raw)['Campus']


@pytest.fixture
def config_simple_all(config_simple_raw):
    """A validated configuration with one {cohort, schedule block, site}."""
    # This fixture leaves the campus at the root level, whereas
    # `config_simple` goes one level down by default.
    return validate_config(config_simple_raw)


@pytest.fixture
def config_two_blocks(config_simple):
    config_simple['policy']['blocks'] = {
        'earlier': {
            'start': ts_parse('08:00:00'),
            'end': ts_parse('12:00:00')
        },  # 4-hour overlap
        'later': {
            'start': ts_parse('12:00:00'),
            'end': ts_parse('22:00:00')
        }  # 8-hour overlap
    }
    return config_simple


@pytest.fixture
def people_simple_raw():
    """A one-person roster defined wrt the `config_simple` fixture."""
    return [{
        'id': 'a',
        'campus': 'Campus',
        'cohort': 'People',
        'schedule': {
            '2020-01-01': ['Block'],
            '2020-01-03': ['Block']
        },
        'site_rank': ['Testing']
    }]


@pytest.fixture
def people_simple(people_simple_raw, config_simple_all):
    """A validated one-person roster defined wrt the `config_simple` fixture"""
    return validate_people(people_simple_raw, config_simple_all)

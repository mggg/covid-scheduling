""""Common fixtures for unit tests."""
import pytest
from datetime import timedelta
from dateutil.parser import parse as ts_parse
from covid_scheduling.constants import DAYS, MIDNIGHT
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


@pytest.fixture
def schedules_by_cohort_one_cohort(config_simple):
    """A listing of schedules (each with one appointment) for one cohort."""
    block = config_simple['policy']['blocks']['Block']
    date = block['start'].replace(**MIDNIGHT)
    return {
        'Cohort': [[{
            'date': date + timedelta(days=day),
            'start': block['start'] + timedelta(days=day),
            'end': block['end'] + timedelta(days=day),
            'block': 'Block',
            'weekday': date.strftime('%A'),
            'site': 'Testing'
        }] for day in range(7)]
    }


@pytest.fixture
def schedules_by_cohort_full_dupes(schedules_by_cohort_one_cohort):
    """Two cohorts with completely redundant schedules."""
    return {
        'Cohort1': schedules_by_cohort_one_cohort['Cohort'],
        'Cohort2': schedules_by_cohort_one_cohort['Cohort']
    }


@pytest.fixture
def schedules_by_cohort_partial_dupes(schedules_by_cohort_one_cohort):
    """Two cohorts with partially redundant chedules."""
    one_indices = [0, 2, 4, 6]  # every other day
    two_indices = [0, 3, 6]  # every three days
    orig = schedules_by_cohort_one_cohort['Cohort']
    return {
        'Cohort1': [orig[idx] for idx in one_indices],
        'Cohort2': [orig[idx] for idx in two_indices]
    }

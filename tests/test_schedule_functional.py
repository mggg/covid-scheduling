"""Functional tests for the scheduling API."""
import pytest
import random
import string
from itertools import product
from dateutil.parser import parse as ts_parse
from covid_scheduling.constants import DAYS
from covid_scheduling.schedule import assign_schedules
from covid_scheduling.schemas import validate_people, validate_config

@pytest.fixture(params=['1 block', '3 blocks', '10 blocks'])
def block_schedule(request):
    return {
        '1 block': {
            'Block': {
                'start': '08:00:00',
                'end': '20:00:00'
            }
        },
        '3 blocks': {
            'Morning': {
                'start': '08:00:00',
                'end': '12:00:00'
            },
            'Afternoon': {
                'start': '12:00:00',
                'end': '16:00:00'
            },
            'Evening': {
                'start': '16:00:00',
                'end': '20:00:00'
            }
        },
        '10 blocks': {
            **{
                str(hour): {
                    'start': '{:02d}:00:00'.format(hour),
                    'end': '{:02d}:00:00'.format(hour + 1)
                } for hour in range(8, 12)
            },  # 4 blocks (8 a.m. – 12 p.m.)
            # Lunch break! (12 p.m. – 1 p.m.)
            **{
                str(hour): {
                    'start': '{:02d}:00:00'.format(hour),
                    'end': '{:02d}:00:00'.format(hour + 1)
                } for hour in range(13, 18)
            },  # 5 blocks (1 p.m. – 6 p.m.)
            **{
                '18': {
                    'start': '18:00:00',
                    'end': '20:00:00'
                }
            }  # 1 double-long block (6 p.m. – 8 p.m.)
        }
    }[request.param]


@pytest.fixture(params=['every day', 'weekdays', 'M/W/F'])
def day_schedule(request):
    return {
        'every day': DAYS,
        'weekdays': DAYS[:-2],
        'M/W/F': ['Monday', 'Wednesday', 'Friday']
    }[request.param]


@pytest.fixture(params=['1 site', '3 sites'])
def sites(request, day_schedule):
    site_days = {
        'A': {
            'n_lines': 3,
            'hours': [
                {'start': '08:00:00', 'end': '20:00:00'}  # all day
            ]
        },
        'B': {  # more lines, fewer hours (vs. A)
            'n_lines': 5,
            'hours': [
                {'start': '09:00:00', 'end': '17:00:00'}
            ]
        },
        'C': {  # irregular hours
            'n_lines': 2,
            'hours': [
                {'start': '08:30:00', 'end': '10:30:00', 'weight': 1.2},
                {'start': '11:00:00', 'end': '14:00:00', 'weight': 0.8},
                {'start': '15:00:00', 'end': '19:00:00', 'weight': 1}
            ]
        }
    }

    if request.param == '1 site':
        sites = {'A': site_days['A']}
    else:
        sites = site_days

    for site, config in sites.items():
        with_days = []
        for day in day_schedule:
            for window in config['hours']:
                with_days.append({'day': day, **window})
        config['hours'] = with_days
    return sites


@pytest.fixture(params=['1x', '2x', '1x/2x'])
def cohorts(request):
    coh = {
        '1x': {
            'interval': {
                'min': 1,
                'target': 7,
                'max': 13
            }
        },
        '2x': {
            'interval': {
                'min': 1,
                'target': 3.5,
                'max': 6
            }
        }
    }
    return {c: coh[c] for c in request.param.split('/')}


@pytest.fixture(params=['none', 'day', 'block', 'day/block'])
def load_balancing(request):
    day_load = {'min': 0.5, 'max': 1.5}
    block_load = {'min': 0.25, 'max': 2}
    return {
        'none': {},
        'day': {'day_load_tolerance': day_load},
        'block': {'block_load_tolerance': block_load},
        'day/block': {
            'day_load_tolerance': day_load,
            'block_load_tolerance': block_load
        }
    }


@pytest.fixture
def campus_config(block_schedule,
                  sites,
                  cohorts,
                  load_balancing):
    return {
        'policy': {
            'blocks': block_schedule,
            'cohorts': cohorts,
            'params': load_balancing
        },
        'sites': sites
    }


@pytest.fixture(params=[
    c for c in product(
        [1, 10, 100, 300, 1000],
        [0.2, 0.5, 0.8]
    )
])
def people(request, block_schedule, cohorts, sites):
    n, p = request.param
    def random_sites(sites):
        if len(sites) == 1:
            return list(sites.keys())
        else:
            all_sites = list(sites.keys())
            random.shuffle(all_sites)
            return all_sites[:2]

    return [
        {
            'id': ''.join(random.choice(string.ascii_lowercase +
                                        string.ascii_uppercase)
                          for _ in range(10)),
            'campus': 'Campus',
            'cohort': random.choice(list(cohorts.keys())),
            'schedule': {
                '2020-01-{:02d}'.format(day): [
                    b for b in block_schedule.keys()
                    if random.random() < p
                ]
                for day in range(1, 8)
            },
            'site_rank': random_sites(sites)
        }
        for _ in range(n)
    ]

def test_scheduler(campus_config, people):
    if random.random() > 0.05:
        pytest.skip('randomly skipping')
    """
    validated_config = validate_config({'Campus': campus_config})
    validated_people = validate_people(people, validated_config)
    assignments, _ = assign_schedules(
        config=validated_config['Campus'],
        people=validated_people,
        start_date=ts_parse('2020-01-01'),
        end_date=ts_parse('2020-01-07')
    )
   """

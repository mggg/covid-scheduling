"""Functional tests for the scheduling API.

We randomly generate configurations and people and verify for each combination
that all high-level scheduling invariants are satisfied. These invariants
include, but are not limited to:
    * All people are assigned only to appointmenst they're available for.
    * All people are assigned only to sites they ranked.
    * Each person has the number of tests they need if such an assignment
      is possible.

These tests are intentionally nondeterministic, though a random seed could be
fixed in the future if necessary. (For each run, we are essentially generating
a stratified random sample from the space of all possible tests.) They
complement the small set of handcrafted test cases in the unit tests---they are
designed to expose unexpected behavior and verify that the API contract is met
over a wide variety of cases that would be difficult to construct by hand.
We currently use the following parameters:
    * Schedule blocks: 1, 3, or 10* blocks per day
    * Site schedules:  open every day, weekdays*, or Monday/Wednesday/Friday
    * Testing sites: 1 or 3 sites
    * Testing cohorts: 1x/week, 2x/week, both
    * Load balancing constraint(s): none, day-level*, block-level*, both
    * Number of people: 1, 10, 100, 250, 1000*, 12,500*
    * Probability that a person is available in a given block:
        0.2 (sparsest schedules), 0.5, 0.8 (densest schedules)
    * Testing window: 7 days

Parameters marked with * are only used in slow mode, which can be activated
with the --run-slow flag when running `pytest`. Slow mode is not necessary
in most cases and may be infeasible in a CI/CD pipeline; it should generally
be used only for stress testing.
"""
import pytest
import random
import string
import warnings
import numpy as np
from itertools import product
from typing import List, Dict, Union
from dateutil.parser import parse as ts_parse
from covid_scheduling.errors import AssignmentError
from covid_scheduling.schedule import assign_schedules
from covid_scheduling.load_balancing import site_weights
from covid_scheduling.constants import DAYS, MIDNIGHT, SEC_PER_DAY
from covid_scheduling.schemas import validate_people, validate_config

START_DATE = ts_parse('2020-01-01')
END_DATE = ts_parse('2020-01-07')


@pytest.fixture(params=[
    '1 block', '3 blocks',
    pytest.param('10 blocks', marks=pytest.mark.slow)
])
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
                }
                for hour in range(8, 12)
            },  # 4 blocks (8 a.m. – 12 p.m.)
            # Lunch break! (12 p.m. – 1 p.m.)
            **{
                str(hour): {
                    'start': '{:02d}:00:00'.format(hour),
                    'end': '{:02d}:00:00'.format(hour + 1)
                }
                for hour in range(13, 18)
            },  # 5 blocks (1 p.m. – 6 p.m.)
            **{
                '18': {
                    'start': '18:00:00',
                    'end': '20:00:00'
                }
            }  # 1 double-long block (6 p.m. – 8 p.m.)
        }
    }[request.param]


@pytest.fixture(params=[
    'every day',
    pytest.param('weekdays', marks=pytest.mark.slow), 'M/W/F'
])
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
            'hours': [{
                'start': '08:00:00',
                'end': '20:00:00'
            }  # all day
                      ]
        },
        'B': {  # more lines, fewer hours (vs. A)
            'n_lines': 5,
            'hours': [{
                'start': '09:00:00',
                'end': '17:00:00'
            }]
        },
        'C': {  # irregular hours
            'n_lines':
            2,
            'hours': [{
                'start': '08:30:00',
                'end': '10:30:00',
                'weight': 1.2
            }, {
                'start': '11:00:00',
                'end': '14:00:00',
                'weight': 0.8
            }, {
                'start': '15:00:00',
                'end': '19:00:00',
                'weight': 1
            }]
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
    if '1x' in request.param:
        fallback_2x = ['1x']
    else:
        fallback_2x = []
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
            },
            'fallback': fallback_2x
        }
    }
    return {c: coh[c] for c in request.param.split('/')}


@pytest.fixture(params=[
    'none',
    pytest.param('day', marks=pytest.mark.slow),
    pytest.param('block', marks=pytest.mark.slow), 'day/block'
])
def load_balancing(request):
    day_load = {'min': 0.5, 'max': 1.5}
    block_load = {'min': 0.25, 'max': 2}
    return {
        'none': {},
        'day': {
            'day_load_tolerance': day_load
        },
        'block': {
            'block_load_tolerance': block_load
        },
        'day/block': {
            'day_load_tolerance': day_load,
            'block_load_tolerance': block_load
        }
    }[request.param]


@pytest.fixture
def campus_config(block_schedule, sites, cohorts, load_balancing):
    return {
        'policy': {
            'blocks': block_schedule,
            'cohorts': cohorts,
            'params': load_balancing
        },
        'sites': sites
    }


@pytest.fixture(params=[
    1, 10, 100, 250,
    pytest.param(1000, marks=pytest.mark.slow),
    pytest.param(12500, marks=pytest.mark.slow)
])
def n_people(request):
    return request.param


@pytest.fixture(params=[0.2, 0.5, 0.8])
def block_prob(request):
    return request.param


@pytest.fixture
def people(n_people, block_prob, block_schedule, cohorts, sites):
    def random_sites(sites):
        if len(sites) == 1:
            return list(sites.keys())
        else:
            all_sites = list(sites.keys())
            random.shuffle(all_sites)
            return all_sites[:2]

    p = [{
        'id':
        ''.join(
            random.choice(string.ascii_lowercase + string.ascii_uppercase)
            for _ in range(10)),
        'campus':
        'Campus',
        'cohort':
        random.choice(list(cohorts.keys())),
        'schedule': {
            '2020-01-{:02d}'.format(day):
            [b for b in block_schedule.keys() if random.random() < block_prob]
            for day in range(1, 8)
        },
        'site_rank':
        random_sites(sites)
    } for _ in range(n_people)]
    return p


def people_available(people_before: List, people_after: List) -> List[bool]:
    """Verifies that each person is available during their assigned schedule.

    Args:
        people_before: A roster of unscheduled people.
        people_after: The same roster of people with assigned schedules.
            (Order not required to match `people_before`.)

    Returns:
        A list of booleans indicating whether each person (in the order
        of `people_before`) is only scheduled during blocks they are available
        in.  Note that if a person is not scheduled at all or not available in
        `people_after`, their schedule is considered valid by this criterion.
    """
    people_after_by_id = {p['id']: p for p in people_after}
    valid = []
    for person_before in people_before:
        person_after = people_after_by_id[person_before['id']]
        schedule_before = set()
        schedule_after = set()
        if 'schedule' in person_before and 'schedule' in person_after:
            for date, blocks in person_before['schedule'].items():
                for block in blocks:
                    schedule_before.add((date, block))
            for date, blocks in person_after['schedule'].items():
                for block in blocks:
                    schedule_after.add((ts_parse(date), block['block']))
            valid.append(not bool(schedule_after - schedule_before))
        else:
            valid.append(True)
    return valid


def appointments_valid(people_after: List, config: Dict) -> List[bool]:
    """Verifies that all appointments are assigned to open testing stations.

    Args:
        people_after: A roster of people with assigned schedules.
        config: A campus-level configuration.

    Returns:
        A list of booleans indicating whether each person (in the order
        of `people_before`) has a list of feasible appointments---that is,
        the testing station of each appointment is open during at least
        part of the appointment's time block.
    """
    def person_valid(person):
        for date, blocks in person['schedule'].items():
            weekday = ts_parse(date).strftime('%A')
            for block in blocks:
                name = block['block']
                site = block['site']
                start = config['policy']['blocks'][name]['start']
                end = config['policy']['blocks'][name]['end']
                window_match = False
                for window in config['sites'][site]['hours']:
                    overlap_start = max(start, window['start'])
                    overlap_end = min(end, window['end'])
                    if (window['day'] == weekday and
                        (overlap_end - overlap_start).total_seconds() > 0):
                        window_match = True
                        break
                if not window_match:
                    return False
        return True

    return [person_valid(person) for person in people_after]


def demand_satisfied(people_after: List) -> List[bool]:
    """Verifies that each person gets the appropriate number of appointments.

    We assume that scheduling occurs over a single week. Thus, people in the
    `1x` cohort get one test, people in the `2x` cohort get two tests,
    and people without a cohort due to incompatibility (the `None` cohort)
    get no tests. All people in the `None` cohort should not be `assigned`;
    people in the other cohorts should be `assigned`.

    Args:
        people_after: A roster of people with assigned schedules.

    Returns:
        A list of booleans indicating whether each person has the right number
        of appointments.
    """
    valid = []
    cohort_demands = {'1x': 1, '2x': 2, None: 0}
    for person in people_after:
        n_appointments = sum(len(day) for day in person['schedule'].values())
        if person['assigned']:
            valid.append(n_appointments == cohort_demands[person['cohort']]
                         and person['cohort'] is not None)
        else:
            valid.append(n_appointments == 0 and person['cohort'] is None)
    return valid


def _intervals(person: Dict, config: Dict) -> List[float]:
    """Returns all start-to-start intervals in a person's schedule.

    Args:
        person: The person's data.
        config: The configuration of the person's campus.

    Returns:
        A list of the intervals (in days) between blocks in the
        person's schedule.
    """
    # TODO: consider history!!!
    block_starts = {
        block: data['start'] - data['start'].replace(**MIDNIGHT)
        for block, data in config['policy']['blocks'].items()
    }
    starts = []
    for date, blocks in person['schedule'].items():
        for block in blocks:
            starts.append(ts_parse(date) + block_starts[block['block']])
    sorted_starts = sorted(starts)
    deltas = []
    for left, right in zip(starts[:-1], starts[1:]):
        deltas.append((right - left).total_seconds() / SEC_PER_DAY)
    return deltas


def _valid_for_cohort(person: Dict, coh: Union[str, None],
                      config: Dict) -> bool:
    """Determines whether a person's schedule is compatible with a cohort.

    Args:
        person: The person's data.
        coh: The name of the cohort to check (or `None`).
        config: The configuration of the person's campus.

    Returns:
        Whether the person is compatible with the cohort.
    """
    if coh is None:
        # The `None` cohort allows any number of blocks at any interval.
        return True
    min_days = config['policy']['cohorts'][coh]['interval']['min']
    max_days = config['policy']['cohorts'][coh]['interval']['max']
    intervals = _intervals(person, config)
    return not intervals or (min(intervals) >= min_days
                             and max(intervals) <= max_days)


def intervals_valid(people_after: List, config: Dict) -> List[bool]:
    """Verifies that each person is tested at a valid testing interval.

    Args:
        people_after: A roster of people with assigned schedules.
        config: A campus-level configuration.

    Returns:
        A list of booleans indicating whether each person is compatible
        with their cohort.
    """
    return [_valid_for_cohort(p, p['cohort'], config) for p in people_after]


def fallback_valid(people_before: List, people_after: List,
                   config: Dict) -> List[bool]:
    """Verifies that fallback cohorts are assigned correctly, if applicable.

    If there is only one cohort in the configuration, the only two cohorts in
    the output should be that cohort and `None`. A person should be in the
    `None` cohort iff their schedule does not allow them to be scheduled within
    their original cohort's bounds.

    If there are two cohorts (1x/2x) in the configuration, then people in the
    2x cohort should be assigned to the 1x cohort iff they are compatible with
    1x but not 2x. People compatible with neither should be assigned to `None`.

    It is difficult to check that there are _no_ schedules for a particular
    cohort that a person is compatible with without calling a myriad of
    internal functions, so instead simply we check for ordering: a person
    in 1x can only be moved to `None`, and a person in 2x can be moved
    to either 1x or `None`.

    Args:
        people_before: A roster of unscheduled people.
        people_after: The same roster of people with assigned schedules.
            (Order not required to match `people_before`.)
        config: A campus-level configuration.

    Returns:
        A list of booleans indicating whether each person (in the order
        of `people_before`) has a properly assigned cohort.
    """
    people_after_by_id = {p['id']: p for p in people_after}
    valid = []
    cohorts = config['policy']['cohorts']
    full_hierarchy = ['2x', '1x']
    hierarchy = [h for h in full_hierarchy if h in cohorts] + [None]
    for person_before in people_before:
        coh_before = person_before['cohort']
        person_after = people_after_by_id[person_before['id']]
        coh_after = person_after['cohort']
        valid.append(hierarchy.index(coh_after) >= hierarchy.index(coh_before))
    return valid


def load_balanced(people_after: List, config: Dict) -> bool:
    """Verifies that a schedule is load-balanced within specified tolerances.

    Args:
        people_after: A roster of people with assigned schedules.
        config: A campus-level configuration.

    Returns:
        A boolean indicating whether the overall schedule is load-balanced
        according to the configuration.
    """
    # We assume that `site_weights` yields the proper load
    # balancing weights.
    block_starts = {
        block: data['start'] - data['start'].replace(**MIDNIGHT)
        for block, data in config['policy']['blocks'].items()
    }
    for param in ('day_load_tolerance', 'block_load_tolerance'):
        if param in config['policy']['params']:
            weights, site_time_ids = site_weights(
                config=config,
                start_date=START_DATE,
                end_date=END_DATE,
                use_days=(param == 'day_load_tolerance'))
            min_load = config['policy']['params'][param]['min']
            max_load = config['policy']['params'][param]['max']
            counts = np.zeros_like(weights)
            for person in people_after:
                for date, blocks in person['schedule'].items():
                    for block in blocks:
                        if param == 'day_load_tolerance':
                            st_id = site_time_ids[(ts_parse(date),
                                                   block['site'])]
                        else:
                            st_id = site_time_ids[(
                                ts_parse(date) + block_starts[block['block']],
                                block['site'])]
                        counts[st_id] += 1
            counts /= counts.sum()
            if not (np.all(min_load * weights <= counts)
                    and np.all(counts <= max_load * weights)):
                return False
        return True


@pytest.mark.functional
def test_scheduler(campus_config, people):
    """Scheduler invariant checks.

    For a randomly generated configuration and roster of people, we verify:
        * Everyone is scheduled only when they are available.
        * Everyone has the number of appointments required by their cohort,
          unless they have been assigned to a fallback cohort.
        * Everyone with more than one appointment has a testing interval
          allowed by their cohort.
        * Everyone's appointments are valid: for each appointment, the site
          they are assigned to is open during the appointment.
    """
    is_load_balanced = (
        'day_load_tolerance' in campus_config['policy']['params']
        or 'block_load_tolerance' in campus_config['policy']['params'])

    if random.random() > 0.25:
        pytest.skip('randomly skipping')
    elif len(people) < 100 and is_load_balanced:
        pytest.skip('ignoring load balancing for small number of people')

    validated_config = validate_config({'Campus': campus_config})
    validated_campus_conf = validated_config['Campus']
    people_before = validate_people(people, validated_config)

    try:
        people_after, _ = assign_schedules(config=validated_config,
                                           people=people_before,
                                           start_date=START_DATE,
                                           end_date=END_DATE)
    except AssignmentError:
        if is_load_balanced:
            # Try loosening the load balancing constraints to something
            # incredibly loose and re-running.
            warnings.warn('First attempt failed. Loosening load balancing '
                          'constraints and re-running.')
            for param in ('day_load_tolerance', 'block_load_tolerance'):
                if param in validated_config['Campus']['policy']['params']:
                    validated_config['Campus']['policy']['params'][param] = {
                        'min': 0,
                        'max': 100
                    }
            people_after, _ = assign_schedules(config=validated_config,
                                               people=people_before,
                                               start_date=START_DATE,
                                               end_date=END_DATE)
        else:
            pytest.fail('Uncaught AssignmentError')

    assert all(people_available(people_before, people_after))
    assert all(appointments_valid(people_after, validated_campus_conf))
    assert all(demand_satisfied(people_after))
    assert all(intervals_valid(people_after, validated_campus_conf))
    assert all(
        fallback_valid(people_before, people_after, validated_campus_conf))
    assert load_balanced(people_after, validated_campus_conf)

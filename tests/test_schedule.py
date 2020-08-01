"""Unit tests for scheduling utility functions."""
import pytest
import numpy as np
from collections import defaultdict
from dateutil.parser import parse as ts_parse
from covid_scheduling.constants import DAYS
from covid_scheduling.schedule import (schedule_blocks, cohort_schedules,
                                       add_sites_to_schedules, schedule_cost,
                                       schedule_ordering, cohort_tests,
                                       format_assignments, MAX_TESTS)
# Avoid pytest conflicts.
from covid_scheduling.schedule import testing_demand as demand_for_tests

EPS = 1e-8
SECS_PER_DAY = 60 * 60 * 24


def test_schedule_blocks_single_block_day(config_simple):
    day = ts_parse('2020-01-01')
    blocks = schedule_blocks(config_simple, day, day)
    assert blocks == [{
        'date': ts_parse('2020-01-01'),
        'weekday': 'Wednesday',
        'block': 'Block',
        'start': ts_parse('2020-01-01T08:00:00'),
        'end': ts_parse('2020-01-01T20:00:00'),
    }]


def test_schedule_blocks_single_block_week(config_simple):
    start_date = ts_parse('2020-01-01')
    end_date = ts_parse('2020-01-07')
    blocks = schedule_blocks(config_simple, start_date, end_date)
    weekdays = DAYS[2:] + DAYS[:2]
    assert blocks == [{
        'date': ts_parse('2020-01-{:02d}'.format(day)),
        'weekday': weekday,
        'block': 'Block',
        'start': ts_parse('2020-01-{:02d}T08:00:00'.format(day)),
        'end': ts_parse('2020-01-{:02d}T20:00:00'.format(day))
    } for day, weekday in zip(range(1, 8), weekdays)]


def test_schedule_blocks_two_block_day(config_two_blocks):
    day = ts_parse('2020-01-01')
    blocks = schedule_blocks(config_two_blocks, day, day)
    assert blocks == [{
        'date': ts_parse('2020-01-01'),
        'weekday': 'Wednesday',
        'block': 'earlier',
        'start': ts_parse('2020-01-01T08:00:00'),
        'end': ts_parse('2020-01-01T12:00:00')
    }, {
        'date': ts_parse('2020-01-01'),
        'weekday': 'Wednesday',
        'block': 'later',
        'start': ts_parse('2020-01-01T12:00:00'),
        'end': ts_parse('2020-01-01T22:00:00')
    }]


def test_cohort_schedules_single_block_day_one_test(config_simple):
    day = ts_parse('2020-01-01')
    blocks = schedule_blocks(config_simple, day, day)
    schedules = cohort_schedules(config_simple, 'People', 1, blocks)
    assert schedules == [({
        'date': ts_parse('2020-01-01'),
        'weekday': 'Wednesday',
        'block': 'Block',
        'start': ts_parse('2020-01-01T08:00:00'),
        'end': ts_parse('2020-01-01T20:00:00'),
    }, )]


def test_cohort_schedules_single_block_day_two_tests(config_simple):
    day = ts_parse('2020-01-01')
    blocks = schedule_blocks(config_simple, day, day)
    assert cohort_schedules(config_simple, 'People', 2, blocks) == []


def test_cohort_schedules_single_block_week_one_test(config_simple):
    start_date = ts_parse('2020-01-01')
    end_date = ts_parse('2020-01-07')
    blocks = schedule_blocks(config_simple, start_date, end_date)
    schedules = cohort_schedules(config_simple, 'People', 1, blocks)
    weekdays = DAYS[2:] + DAYS[:2]
    assert schedules == [({
        'date':
        ts_parse('2020-01-{:02d}'.format(day)),
        'weekday':
        weekday,
        'block':
        'Block',
        'start':
        ts_parse('2020-01-{:02d}T08:00:00'.format(day)),
        'end':
        ts_parse('2020-01-{:02d}T20:00:00'.format(day))
    }, ) for day, weekday in zip(range(1, 8), weekdays)]


def test_cohort_schedules_single_block_week_two_tests(config_simple):
    start_date = ts_parse('2020-01-01')
    end_date = ts_parse('2020-01-07')
    blocks = schedule_blocks(config_simple, start_date, end_date)
    schedules = cohort_schedules(config_simple, 'People', 2, blocks)
    assert len(schedules) == 21  # 7 choose 2
    for schedule in schedules:
        assert len(schedule) == 2
        start_delta = schedule[1]['start'] - schedule[0]['start']
        assert SECS_PER_DAY <= start_delta.total_seconds() <= 6 * SECS_PER_DAY


def test_cohort_schedules_two_block_week_two_tests(config_two_blocks):
    start_date = ts_parse('2020-01-01')
    end_date = ts_parse('2020-01-07')
    blocks = schedule_blocks(config_two_blocks, start_date, end_date)
    schedules = cohort_schedules(config_two_blocks, 'People', 2, blocks)
    assert len(schedules) == 77  # 14 choose 2 - all pairings with <1 day gap
    for schedule in schedules:
        assert len(schedule) == 2
        start_delta = schedule[1]['start'] - schedule[0]['start']
        assert SECS_PER_DAY <= start_delta.total_seconds() <= 6 * SECS_PER_DAY


def test_cohort_schedules_two_block_week_tight_interval(config_two_blocks):
    config_two_blocks['policy']['cohorts']['People']['interval'] = {
        'min': 3,
        'target': 3.5,
        'max': 4
    }
    start_date = ts_parse('2020-01-01')
    end_date = ts_parse('2020-01-07')
    blocks = schedule_blocks(config_two_blocks, start_date, end_date)
    schedules = cohort_schedules(config_two_blocks, 'People', 2, blocks)
    # * 8 schedules with 3-day interval
    # * 7 schedules with ~3.5-day interval
    # * 6 schedules with 3-day interval
    assert len(schedules) == 21
    interval_counts = defaultdict(int)
    for schedule in schedules:
        assert len(schedule) == 2
        start_delta = schedule[1]['start'] - schedule[0]['start']
        delta_sec = start_delta.total_seconds()
        interval_counts[delta_sec / SECS_PER_DAY] += 1
        assert 3 * SECS_PER_DAY <= delta_sec <= 4 * SECS_PER_DAY
    min_interval = min(interval_counts.keys())
    max_interval = max(interval_counts.keys())
    assert np.abs(min_interval - 3) < EPS
    assert np.abs(max_interval - 4) < EPS
    assert interval_counts[min_interval] == 8
    assert interval_counts[max_interval] == 6


def test_add_sites_to_schedules_single_block_day_one_test(config_simple):
    day = ts_parse('2020-01-01')
    blocks = schedule_blocks(config_simple, day, day)
    schedules = cohort_schedules(config_simple, 'People', 1, blocks)
    with_sites = add_sites_to_schedules(schedules, config_simple)
    assert with_sites == [[
        {
            'site': 'Testing',
            'date': ts_parse('2020-01-01'),
            'weekday': 'Wednesday',
            'block': 'Block',
            'start': ts_parse('2020-01-01T08:00:00'),
            'end': ts_parse('2020-01-01T20:00:00'),
        },
    ]]


def _by_s(v):
    return v[0]['start']


def test_schedule_ordering_one_cohort(schedules_by_cohort_one_cohort):
    by_id, with_id = schedule_ordering(schedules_by_cohort_one_cohort)
    assert (sorted(by_id, key=_by_s) == sorted(
        schedules_by_cohort_one_cohort['Cohort'], key=_by_s))
    assert len(with_id['Cohort']) == len(
        schedules_by_cohort_one_cohort['Cohort'])
    for sched in with_id['Cohort']:
        assert sched['blocks'] == by_id[sched['id']]


def test_schedule_ordering_full_dupes(schedules_by_cohort_full_dupes):
    by_id, with_id = schedule_ordering(schedules_by_cohort_full_dupes)
    assert (sorted(by_id, key=_by_s) == sorted(
        schedules_by_cohort_full_dupes['Cohort1'], key=_by_s) == sorted(
            schedules_by_cohort_full_dupes['Cohort2'], key=_by_s))
    for cohort in ('Cohort1', 'Cohort2'):
        for sched in with_id[cohort]:
            assert sched['blocks'] == by_id[sched['id']]


def test_schedule_ordering_partial_dupes(schedules_by_cohort_partial_dupes):
    by_id, with_id = schedule_ordering(schedules_by_cohort_partial_dupes)
    co1_start = set(s[0]['start']
                    for s in schedules_by_cohort_partial_dupes['Cohort1'])
    co2_start = set(s[0]['start']
                    for s in schedules_by_cohort_partial_dupes['Cohort2'])
    assert set(s[0]['start'] for s in by_id) == co1_start.union(co2_start)
    for cohort in ('Cohort1', 'Cohort2'):
        for sched in with_id[cohort]:
            assert sched['blocks'] == by_id[sched['id']]


def test_cohort_tests():
    config = {
        'policy': {
            'cohorts': {
                'every day': {
                    'interval': {
                        'target': 1
                    }
                },
                'every 2 days': {
                    'interval': {
                        'target': 2
                    }
                },
                'every 3 days': {
                    'interval': {
                        'target': 3
                    }
                },
                'twice a week': {
                    'interval': {
                        'target': 3.5
                    }
                },
                'every week': {
                    'interval': {
                        'target': 7
                    }
                },
                'every 2 weeks': {
                    'interval': {
                        'target': 14
                    }
                }
            }
        }
    }
    test_counts = cohort_tests(config, 7)
    assert test_counts == {
        'every day': min(MAX_TESTS, 7),
        'every 2 days': min(MAX_TESTS, 4),
        'every 3 days': min(MAX_TESTS, 2),
        'twice a week': min(MAX_TESTS, 2),
        'every week': 1,
        'every 2 weeks': 1
    }


def test_testing_demand_one_cohort(config_simple, people_simple):
    extended_people = 5 * people_simple
    demand = demand_for_tests(config_simple, extended_people, {'People': 3})
    assert np.all(demand == 3 * np.ones(5, dtype=np.int))


def test_testing_demand_two_cohorts(config_simple, people_simple):
    person_one = people_simple[0]
    person_two = person_one.copy()
    person_two['cohort'] = 'People2'
    extended_people = (5 * [person_one]) + (5 * [person_two])
    expected = np.ones(10, dtype=np.int)
    expected[:5] = 3
    expected[5:] = 4
    demand = demand_for_tests(config_simple, extended_people, {
        'People': 3,
        'People2': 4
    })
    assert np.all(expected == demand)


def test_format_assignments_one_person(schedules_by_id_one_cohort,
                                       people_simple):
    assignments = format_assignments(people_simple, schedules_by_id_one_cohort,
                                     {0: 0})
    person = people_simple[0]
    schedule = schedules_by_id_one_cohort[0]
    by_date = {
        schedule[0]['date'].strftime('%Y-%m-%d'): [{
            'block':
            schedule[0]['block'],
            'site':
            schedule[0]['site']
        }]
    }
    assert assignments == [{
        'id': person['id'],
        'cohort': person['cohort'],
        'campus': person['campus'],
        'assigned': True,
        'schedule': by_date
    }]


def test_format_assignments_one_person_error(schedules_by_id_one_cohort,
                                             people_simple):
    assignments = format_assignments(people_simple, schedules_by_id_one_cohort,
                                     {0: None})
    person = people_simple[0]
    assert assignments == [{
        'id': person['id'],
        'cohort': person['cohort'],
        'campus': person['campus'],
        'assigned': False,
        'error': 'Not enough availability.',
        'schedule': {}
    }]


def test_format_assignments_one_person_missing(schedules_by_id_one_cohort,
                                               people_simple):
    assignments = format_assignments(people_simple, schedules_by_id_one_cohort,
                                     {})
    assert assignments == []


def test_format_assignments_multiple_people(schedules_by_id_one_cohort,
                                            people_simple):
    person_one = people_simple[0].copy()
    person_two = people_simple[0].copy()
    person_two['id'] = 'b'
    people = [person_one, person_two]
    schedules_by_date = [{
        schedule[0]['date'].strftime('%Y-%m-%d'): [{
            'block':
            schedule[0]['block'],
            'site':
            schedule[0]['site']
        }]
    } for schedule in schedules_by_id_one_cohort[:2]]
    assignments = format_assignments(people, schedules_by_id_one_cohort, {
        0: 0,
        1: 1
    })
    assert assignments == [{
        'id': person['id'],
        'cohort': person['cohort'],
        'campus': person['campus'],
        'assigned': True,
        'schedule': schedule
    } for person, schedule in zip(people, schedules_by_date)]

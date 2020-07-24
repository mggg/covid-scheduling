"""Unit tests for scheduling utility functions."""
import pytest
import numpy as np
from collections import defaultdict
from dateutil.parser import parse as ts_parse
from covid_scheduling.constants import DAYS
from covid_scheduling.schedule import (schedule_blocks, cohort_schedules,
                                       add_sites_to_schedules, schedule_cost)

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


# TODO: More of add_sites_to_schedules(), compatible()

"""Campus block scheduling algorithms."""
from typing import Dict, List, Tuple, Iterable
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations, product
import numpy as np  # type: ignore
from covid_scheduling.errors import AssignmentError
from covid_scheduling.bipartite import bipartite_assign
from covid_scheduling.constants import DAYS, SEC_PER_DAY, MIDNIGHT

MAX_DAYS = 14
MAX_TESTS = 2
ASSIGNMENT_METHODS = {'bipartite': bipartite_assign}


def schedule_cost(schedule: List[Dict], person: Dict,
                  target_interval: float) -> float:
    """Computes the cost (spacing + site) of a schedule.

    Currently, we use the sum of squared deviations from the target
    testing interval.
    """
    # TODO: Site ranking costs.
    # TODO: Spacing history costs.
    cost = 0
    for left_b, right_b in zip(schedule[:-1], schedule[1:]):
        delta_sec = (right_b['start'] - left_b['start']).total_seconds()
        cost += ((delta_sec / SEC_PER_DAY) - target_interval)**2
    return cost


def assign_schedules(config: Dict,
                     people: List,
                     start_date: datetime,
                     end_date: datetime,
                     method: str = 'bipartite') -> Tuple[List, List]:
    """Assigns people to testing schedules.

    Args:
        config: A validated university-level configuration.
        people: A validated roster of people.
        start_date: The first day in the assignment range.
        end_date: The first day in the assignment range (inclusive).
        method: The assignment algorithm to use (only bipartite
            matching is available; heuristic algorithms may be
            added in the future).

    Returns:
        Testing schedules for each person (with warnings).
    """
    n_days = (end_date - start_date).days + 1
    if n_days >= MAX_DAYS:
        raise AssignmentError('Could not generate assignments over'
                              f'a {n_days}-day window. Maximum'
                              f'assignment window is {MAX_DAYS} days.')
    if n_days < 0:
        raise AssignmentError('Assignment window starts before it ends.')

    # Filter personal schedules.
    people = deepcopy(people)
    for p in people:
        p['schedule'] = {
            date: blocks for date, blocks in p['schedule'].items()
                         if start_date <= date <= end_date
        }

    # Generate assignments for each campus individually.
    assignments = []
    all_stats = []
    for campus, campus_config in config.items():
        # Enumerate schedules by cohort.
        # We assume that each person demands at least one test, regardless
        # of their cohort's target interval--otherwise, there would be little
        # point in requesting a new schedule. We estimate the number of tests
        # for each cohort using the cohort's target interval, applying an upper
        # bound of `MAX_TESTS` to avoid combinatorial explosion in
        # adversarial cases.
        n_tests = {
            name: min(max(int(round(n_days / data['interval']['target'])), 1),
                      MAX_TESTS)
            for name, data in campus_config['policy']['cohorts'].items()
        }

        blocks = schedule_blocks(campus_config, start_date, end_date)
        schedules_by_cohort = {
            co: add_sites_to_schedules(
                cohort_schedules(campus_config, co, n_tests[co], blocks),
                campus_config)
            for co in campus_config['policy']['cohorts'].keys()
        }
        schedules, schedules_by_cohort = schedule_ordering(schedules_by_cohort)
        campus_people = [p for p in people if p['campus'] == campus]
        demand = testing_demand(campus_config, campus_people, n_tests)
        try:
            assign_fn = ASSIGNMENT_METHODS[method]
        except ValueError:
            raise AssignmentError(f'Assignment method "{method}" '
                                  'not available.')

        condensed, stats = assign_fn(config=campus_config,
                                     people=campus_people,
                                     start_date=start_date,
                                     end_date=end_date,
                                     schedules=schedules,
                                     schedules_by_cohort=schedules_by_cohort,
                                     test_demand=demand,
                                     cost_fn=schedule_cost)
        assignments += format_assignments(people, schedules, condensed)
        all_stats += stats
    return assignments, all_stats


def format_assignments(people: List, schedules: Dict,
                       assignments: Dict) -> List:
    """Converts an assignment map to the proper JSON schema."""
    person_assignments = []
    for person_idx, schedule_idx in assignments.items():
        person = people[person_idx]
        assignment = {
            'id': person['id'],
            'cohort': person['cohort'],
            'campus': person['campus']
        }
        if schedule_idx is None:
            assignment['assigned'] = False
            assignment['error'] = 'Not enough availability.'
        else:
            schedule = schedules[schedule_idx]
            schedule_by_date = defaultdict(list)
            for block in schedule:
                block_date = block['date'].strftime('%Y-%m-%d')
                schedule_by_date[block_date].append({
                    'site': block['site'],
                    'block': block['block']
                })
            assignment['assigned'] = True
            assignment['schedule'] = dict(schedule_by_date)
        person_assignments.append(assignment)
    return person_assignments


def schedule_blocks(config: Dict, start_date: datetime,
                    end_date: datetime) -> List:
    """Enumerates schedule blocks in a date window."""
    blocks = []
    window_days = (end_date - start_date).days + 1
    for day in range(window_days):
        ts = start_date + timedelta(days=day)
        for name, block in config['policy']['blocks'].items():
            start_delta = block['start'] - block['start'].replace(**MIDNIGHT)
            end_delta = block['end'] - block['end'].replace(**MIDNIGHT)
            blocks.append({
                'date': ts,
                'weekday': ts.strftime('%A'),
                'block': name,
                'start': ts + start_delta,
                'end': ts + end_delta
            })
    return sorted(blocks, key=lambda b: b['start'])


def cohort_schedules(config: Dict, cohort: str, n_tests: int,
                     blocks: List) -> List:
    """Enumerates possible schedules for a cohort."""
    min_interval_sec = (SEC_PER_DAY *
                        config['policy']['cohorts'][cohort]['interval']['min'])
    max_interval_sec = (SEC_PER_DAY *
                        config['policy']['cohorts'][cohort]['interval']['max'])
    schedules = []
    for combo in combinations(blocks, n_tests):
        valid = True
        # Compute intervals between adjacent pairs in the proposed schedule.
        for left_b, right_b in zip(combo[:-1], combo[1:]):
            delta_sec = (right_b['start'] - left_b['start']).total_seconds()
            if delta_sec < min_interval_sec or delta_sec > max_interval_sec:
                valid = False
                break
        if valid:
            schedules.append(combo)
    return schedules


def add_sites_to_schedules(schedules: List[Dict], config: Dict) -> List[List]:
    """Augments a list of schedules with site permutations."""
    site_schedules = []
    sites = config['sites'].keys()
    default_day = {'year': 2020, 'month': 9, 'day': 1}
    allow_splits = config.get('bounds', {}).get('allow_site_splits', False)
    for schedule in schedules:
        if allow_splits:
            # 💥 Warning: allowing schedules with multiple appointments
            # to split across sites may result in combinatorial explosion.
            site_iter: Iterable = product(sites, repeat=len(schedule))
        else:
            site_iter = [[site] * len(schedule) for site in sites]
        for site_perm in site_iter:
            augmented = []
            perm_valid = True
            for site, block in zip(site_perm, schedule):
                # Filter out infeasible permutations.
                day_hours = [
                    h for h in config['sites'][site]['hours']
                    if h['day'] == block['weekday']
                ]
                block_start = block['start'].replace(**default_day)
                block_end = block['end'].replace(**default_day)
                overlap = False
                for window in day_hours:
                    window_start = window['start'].replace(**default_day)
                    window_end = window['end'].replace(**default_day)
                    overlap_start = max(block_start, window_start)
                    overlap_end = min(block_end, window_end)
                    if (overlap_end - overlap_start).total_seconds() > 0:
                        overlap = True
                        break
                if overlap:
                    augmented.append({'site': site, **block})
                else:
                    perm_valid = False
                    break
            if perm_valid:
                site_schedules.append(augmented)
    return site_schedules


def schedule_ordering(schedules_by_cohort: Dict) -> Tuple[Dict, Dict]:
    """Assigns a canonical ordering to unique schedules across cohorts."""
    schedule_ids: Dict[Tuple, int] = {}
    schedules_by_id = {}
    schedules_by_cohort_with_id = {}
    sched_id = 0
    for cohort, schedules in schedules_by_cohort.items():
        with_ids = []
        for schedule in schedules:
            uniq = []
            for block in schedule:
                block_hash = (block['start'], block['end'], block['site'])
                uniq += list(block_hash)
            uniq_hash = tuple(uniq)
            if uniq_hash in schedule_ids:
                with_ids.append({
                    'id': schedule_ids[uniq_hash],
                    'blocks': schedule
                })
            else:
                schedule_ids[uniq_hash] = sched_id
                schedules_by_id[sched_id] = schedule
                with_ids.append({'id': sched_id, 'blocks': schedule})
                sched_id += 1
        schedules_by_cohort_with_id[cohort] = with_ids
    return schedules_by_id, schedules_by_cohort_with_id


def testing_demand(config: Dict, people: List, n_tests: Dict) -> np.ndarray:
    """Calculates testing demand for each person."""
    cohort_counts: Dict = defaultdict(int)
    n_people = len(people)
    test_demand = np.zeros(n_people, dtype=np.int)
    for idx, person in enumerate(people):
        cohort_counts[person['cohort']] += 1
        test_demand[idx] = n_tests[person['cohort']]
    return test_demand

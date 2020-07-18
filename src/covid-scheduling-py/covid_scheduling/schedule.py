"""Campus block scheduling algorithms."""
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List
from collections import deque, defaultdict
from datetime import datetime, timedelta
from itertools import combinations, product
import numpy as np
from ortools.sat.python import cp_model
from ortools.linear_solver import pywraplp
from .schemas import DAYS

MAX_DAYS = 14
MAX_TESTS = 2
SEC_PER_DAY = 24 * 60 * 60
INT_SCALE = 1000  # coefficient for rounding cost matrices

def schedule_blocks(config: Dict,
                    start_date: datetime,
                    end_date: datetime) -> List:
    """Enumerates schedule blocks in a date window."""
    blocks = []
    window_days = (end_date - start_date).days + 1
    for day in range(window_days):
        ts = start_date + timedelta(days=day)
        for name, block in config['policy']['blocks'].items():
            midnight = {'hour': 0, 'minute': 0, 'second': 0}
            start_delta = block['start'] - block['start'].replace(**midnight)
            end_delta = block['end'] - block['end'].replace(**midnight)
            blocks.append({
                'date': ts.strftime('%Y-%m-%d'),
                'weekday': ts.strftime('%A'),
                'block': name,
                'start': ts + start_delta,
                'end': ts + end_delta
            })
    return sorted(blocks, key=lambda b: b['start'])


def cohort_schedules(config: Dict,
                     cohort: str,
                     n_tests: int,
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


def add_sites(schedules: List[Dict], config: Dict):
    """Augments a list of schedules with site permutations."""
    site_schedules = []
    sites = config['sites'].keys()
    default_day = {'year': 2020, 'month': 9, 'day': 1}
    for schedule in schedules:
        for site_perm in product(sites, repeat=len(schedule)):
            augmented = []
            perm_valid = True
            for site, block in zip(site_perm, schedule):
                # Filter out infeasible permutations.
                day_hours = [h for h in config['sites'][site]['hours']
                             if h['day'] == block['weekday']]
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


def schedule_cost(schedule: List[Dict], person: Dict, target_interval: float):
    """Computes the cost (spacing + site) of a schedule.

    Currently, we use the sum of squared deviations from the target
    testing interval.
    """
    # TODO: Site ranking costs.
    # TODO: Spacing history costs.
    cost = 0
    for left_b, right_b in zip(schedule[:-1], schedule[1:]):
        delta_sec = (right_b['start'] - left_b['start']).total_seconds()
        cost += ((delta_sec / SEC_PER_DAY) - target_interval) ** 2
    return cost


def compatible(schedule: List[Dict], person: Dict) -> bool:
    """Determines if a person is compatible with a testing schedule."""
    # Constraint: the testing schedule cannot include a site
    # not ranked by the person.
    testing_sites = set(s['site'] for s in schedule)
    person_sites = set(person['site_rank'])
    if testing_sites - person_sites:
        return False

    # Constraint: the person must be available during all
    # blocks of the testing schedule.
    testing_blocks = set((s['date'], s['block']) for s in schedule)
    person_blocks = set()
    for date, blocks in person['schedule'].items():
        ymd = date.strftime('%Y-%m-%d')
        person_blocks = person_blocks.union(set((ymd, b) for b in blocks))
    return not testing_blocks - person_blocks


def site_weights(config: Dict) -> np.ndarray:
    """Determines load-balanced supply for each site-block over a week."""
    blocks = sorted(config['policy']['blocks'].values(), key=lambda k: k['start'])
    sites = config['sites']
    weights = np.zeros((len(DAYS) * len(blocks), len(sites)))
    for day_idx, day in enumerate(DAYS):
        for block_idx, block in enumerate(blocks):
            day_block_idx = (day_idx * len(blocks)) + block_idx
            for site_idx, site in enumerate(sites):
                # Determine seconds of overlap between site availability
                # windows and schedule blocks.
                hours = [s for s in sites[site]['hours'] if s['day'] == day]
                for window in hours:
                    start = max(block['start'], window['start'])
                    end = min(block['end'], window['end'])
                    delta_s = (end - start).total_seconds()
                    if delta_s > 0:
                        weighted_s = (delta_s * window['weight'] *
                                      sites[site]['n_lines'])
                        weights[day_block_idx, site_idx] += weighted_s
    return weights / weights.sum()


def bipartite_assign(config: Dict,
                     people: List,
                     start_date: datetime,
                     end_date: datetime) -> Dict:
    n_people = len(people)
    n_days = (end_date - start_date).days + 1

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
        for name, data in config['policy']['cohorts'].items()
    }

    blocks = schedule_blocks(config, start_date, end_date)
    schedules_by_cohort = {
        co: add_sites(
            cohort_schedules(config, co, n_tests[co], blocks),
            config
        )
        for co in config['policy']['cohorts'].keys()
    }

    # Assign an ordering to all unique schedules.
    schedule_ids = {}
    schedule_ids_by_cohort = {}
    curr_id = 0
    for cohort, schedules in schedules_by_cohort.items():
        with_ids = []
        for schedule in schedules:
            uniq = []
            for block in schedule:
                uniq += [block['start'], block['end'], block['site']]
            uniq = tuple(uniq)
            if uniq in schedule_ids:
                with_ids.append({'id': schedule_ids[uniq], 'blocks': schedule})
            else:
                schedule_ids[uniq] = curr_id
                with_ids.append({'id': curr_id, 'blocks': schedule})
                curr_id += 1
        schedule_ids_by_cohort[cohort] = with_ids

    # Formulate IP minimum-cost matching problem.
    # NOTE: OR-Tools has a specialized linear assignment solver that can be
    # faster than the OR-Tools SAT solver. It may be a good choice if this
    # doesn't scale well; however, it does not allow side constraints,
    # e.g. load balancing. See
    # https://developers.google.com/optimization/assignment/linear_assignment
    model = cp_model.CpModel()
    n_schedules = len(schedule_ids)
    costs = np.zeros((n_people, n_schedules))
    assignments = np.zeros((n_people, n_schedules), dtype=np.int).tolist()
    for p_idx, person in tqdm(enumerate(people)):
        cohort = person['cohort']
        n_matches = 0
        for schedule in schedule_ids_by_cohort[cohort]:
            # A person-schedule assignment variable is only created if
            # the person is compatible with a testing schedule. Thus,
            # the resulting `assignments` matrix is a mix of variables
            # (either 0 or 1; to be determined by the solver)
            # and fixed 0 entries that the solver ignores.
            if (compatible(schedule['blocks'], person) and
                    len(schedule['blocks']) == n_tests[cohort]):
                s_idx = schedule['id']
                assn = model.NewIntVar(0, 1, f'assignments[{p_idx}, {s_idx}]')
                assignments[p_idx][s_idx] = assn
                target = (config['policy']['cohorts'][cohort]
                          ['interval']['target'])
                costs[p_idx, s_idx] = schedule_cost(schedule['blocks'],
                                                    person, target)
                n_matches += 1
        # Constraint: each person has exactly one schedule assignment.
        if n_matches > 0:
            model.Add(sum(assignments[p_idx][j] for j in range(n_schedules)) == 1)

    # Objective: minimize total matching cost.
    # The SAT solver requires integer-valued costs.
    int_costs = np.round(INT_SCALE * costs).astype(np.int)
    model.Minimize(sum([np.dot(a_row, c_row)
                        for (a_row, c_row) in zip(assignments, int_costs)]))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status == cp_model.OPTIMAL:
        print('optimal!')
    elif status == cp_model.FEASIBLE:
        print('feasible.')


    # Add load-balancing constraints (optional).
    """
    n_blocks_per_day = len(config['policy']['blocks'])
    n_sites = len(config['sites'])
    n_site_blocks = n_days * n_blocks_per_day * n_sites

    cohort_counts = defaultdict(int)
    test_demand = np.zeros(n_people, dtype=np.int)
    for idx, person in enumerate(people):
        cohort_counts[person['cohort']] += 1
        test_demand[idx] = n_tests[person['cohort']]
    target_block_load = test_demand.sum() * site_weights(config)

    if 'day_load_tolerance' in config['policy']['bounds']:
        target_day_load = target_block_load.reshape((n_days, n_blocks_per_day,
                                                     n_sites)).sum(axis=1)
    """




def assign_and_adjust(config: Dict,
                      people: List,
                      start_date: datetime,
                      end_date: datetime) -> Dict:
    raise NotImplementedError('Assign-and-adjust algorithm not implemented.')


def assign_schedules(config: Dict,
                     people: List,
                     start_date: datetime,
                     end_date: datetime,
                     method: str = 'bipartite') -> Dict:
    """Assigns people to testing schedules.

    :param config: A validated university-level configuration.
    :param people: A validated roster of people.
    :param start_date: The first day in the assignment range.
    :param end_date: The first day in the assignment range (inclusive).
    :param method: The assignment algorithm to use
                   (`bipartite` or `heuristic`).
    :return: Testing schedules for each person (with warnings).
    """
    window_days = (end_date - start_date).days + 1
    if window_days >= MAX_DAYS:
        raise AssignmentError('Could not generate assignments over'
                              f'a {window_days}-day window. Maximum'
                              f'assignment window is {MAX_DAYS} days.')
    elif window_days < 0:
        raise AssignmentError('Assignment window starts before it ends.')


    # In the future, we might namespace these by university.
    methods = {
        'bipartite': bipartite_assign,
        'heuristic': assign_and_adjust
    }
    if method in methods:
        return methods[method](config, people, start_date, end_date)
    else:
        raise ValueError(f'Assignment method "{method}" does not exist.')


class AssignmentError(Exception):
    """Raised for errors when generating assignments."""

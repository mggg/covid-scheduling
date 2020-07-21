"""Campus block scheduling algorithms."""
from tqdm import tqdm
from copy import deepcopy
from typing import Dict, List
from collections import deque, defaultdict
from datetime import datetime, timedelta
from itertools import combinations, product
import numpy as np
from ortools.linear_solver import pywraplp
from .schemas import DAYS
from .errors import AssignmentError

MAX_DAYS = 14
MAX_TESTS = 2
SEC_PER_DAY = 24 * 60 * 60
MIDNIGHT = {'hour': 0, 'minute': 0, 'second': 0, 'microsecond': 0}

def schedule_blocks(config: Dict,
                    start_date: datetime,
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


def add_sites_to_schedules(schedules: List[Dict], config: Dict):
    """Augments a list of schedules with site permutations."""
    site_schedules = []
    sites = config['sites'].keys()
    default_day = {'year': 2020, 'month': 9, 'day': 1}
    allow_splits = config.get('bounds', {}).get('allow_site_splits', False)
    for schedule in schedules:
        if allow_splits:
            # 💥 Warning: allowing schedules with multiple appointments
            # to split across sites may result in combinatorial explosion.
            site_iter = product(sites, repeat=len(schedule))
        else:
            site_iter = [[site] * len(schedule) for site in sites]
        for site_perm in site_iter:
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


def site_weights(config: Dict,
                 start_date: datetime,
                 end_date: datetime,
                 use_days: bool = False) -> np.ndarray:
    """Determines load-balanced supply for each site-block over a week."""
    n_days = (end_date - start_date).days + 1
    blocks = sorted(config['policy']['blocks'].values(),
                    key=lambda k: k['start'])
    sites = config['sites']
    n_units = len(DAYS)
    if not use_days:
        n_units *= len(blocks)
    weights = np.zeros(len(sites) * n_units)
    site_time_id = 0
    site_time_ids = {}
    for site_idx, site in enumerate(sites):
        for day_idx in range(n_days):
            for block_idx, block in enumerate(blocks):
                ts = start_date + timedelta(days=day_idx)
                weekday = ts.strftime('%A')
                # Determine seconds of overlap between site availability
                # windows and schedule blocks.
                hours = [s for s in sites[site]['hours']
                         if s['day'] == weekday]
                for window in hours:
                    start = max(block['start'], window['start'])
                    end = min(block['end'], window['end'])
                    delta_s = (end - start).total_seconds()
                    if delta_s > 0:
                        weighted_s = (delta_s * window['weight'] *
                                      sites[site]['n_lines'])
                        weights[site_time_id] += weighted_s
                start_d = block['start'] - block['start'].replace(**MIDNIGHT)
                if not use_days:
                    site_time_ids[(ts + start_d, site)] = site_time_id
                    site_time_id += 1
            if use_days:
                site_time_ids[(ts, site)] = site_time_id
                site_time_id += 1
    return weights / weights.sum(), site_time_ids


def bipartite_assign(config: Dict,
                     people: List,
                     start_date: datetime,
                     end_date: datetime) -> Dict:
    """Assigns people to testing schedules.

    :param config: A validated university-level configuration.
    :param people: A validated roster of people.
    :param start_date: The first day in the assignment range.
    :param end_date: The first day in the assignment range (inclusive).
    :return: Testing schedules for each person (with warnings).
    """
    n_people = len(people)
    n_days = (end_date - start_date).days + 1
    if n_days >= MAX_DAYS:
        raise AssignmentError('Could not generate assignments over'
                              f'a {n_days}-day window. Maximum'
                              f'assignment window is {MAX_DAYS} days.')
    if n_days < 0:
        raise AssignmentError('Assignment window starts before it ends.')

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
        co: add_sites_to_schedules(
            cohort_schedules(config, co, n_tests[co], blocks),
            config
        )
        for co in config['policy']['cohorts'].keys()
    }
    schedules, schedules_by_cohort = schedule_ordering(schedules_by_cohort)
    test_demand = testing_demand(config, people, n_tests)

    # Formulate IP minimum-cost matching problem.
    solver = pywraplp.Solver('SolveAssignmentProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    assignments, costs = add_assignments(solver, config, people, schedules,
                                         schedules_by_cohort, test_demand)

    # Add load-balancing constraints (optional).
    if ('day_load_tolerance' in config['policy']['bounds'] or
            'block_load_tolerance' in config['policy']['bounds']):
        # Introduce an auxiliary vector to count schedule occurrences.
        schedule_counts = add_schedule_counts(solver, assignments)
        add_load_balancing(solver, config, people, schedules_by_cohort,
                           schedule_counts, test_demand, start_date, end_date)

    # Objective: minimize total matching cost.
    solver.Minimize(solver.Sum([np.dot(a_row, c_row)
                        for (a_row, c_row) in zip(assignments, costs)]))
    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        condensed = condense_assignments(people, schedules,
                                         assignments)
        return format_assignments(people, schedules, condensed)
    if status == pywraplp.Solver.FEASIBLE:
        raise AssignmentError('Could not generate assignments: '
                              'solution is not optimal.')
    # TODO: are there other statuses?
    raise AssignmentError('Could not generate assignments: '
                          'problem is infeasible.')



def add_assignments(solver, config, people, schedules, schedules_by_cohort,
                    test_demand):
    """Generates assignment and cost matrices."""
    # Cache compatibility sets.
    testing_blocks = {idx: set((s['date'], s['block']) for s in sched)
                      for idx, sched in schedules.items()}
    testing_sites = {idx: set(s['site'] for s in sched)
                      for idx, sched in schedules.items()}
    people_sites = [set(person['site_rank']) for person in people]
    people_blocks = []
    for person in people:
        person_blocks = set()
        for date, blocks in person['schedule'].items():
            for b in blocks:
                person_blocks.add((date, b))
        people_blocks.append(person_blocks)

    # Generate assignment and cost matrices.
    n_people = len(people)
    n_schedules = len(schedules)
    costs = -1 * np.ones((n_people, n_schedules))
    assignments = np.zeros((n_people, n_schedules), dtype=np.int).tolist()
    for p_idx, person in tqdm(enumerate(people)):
        cohort = person['cohort']
        n_matches = 0
        for schedule in schedules_by_cohort[cohort]:
            # A person-schedule assignment variable is only created if
            # the person is compatible with a testing schedule. Thus,
            # the resulting `assignments` matrix is a mix of variables
            # (either 0 or 1; to be determined by the solver)
            # and fixed 0 entries that the solver ignores.
            #
            # A testing schedule and a personal schedule are said to be
            # compatible when:
            #  * The number of tests demanded by the person (determined
            #    by their cohort) matches the number of tests in the
            #    testing schedule.
            #  * There are no blocks in the testing schedule for which
            #    the person is unavailable.
            #  * There are no testing sites in the testing schedule
            #    that the person did not rank.
            s_idx = schedule['id']
            if (len(schedule['blocks']) == test_demand[s_idx] and
                    not testing_blocks[s_idx] - people_blocks[p_idx] and
                    not testing_sites[s_idx]  - people_sites[p_idx]):
                assn = solver.IntVar(0, 1, f'assignments[{p_idx}, {s_idx}]')
                assignments[p_idx][s_idx] = assn
                target = (config['policy']['cohorts'][cohort]
                          ['interval']['target'])
                costs[p_idx, s_idx] = schedule_cost(schedule['blocks'],
                                                    person, target)
                n_matches += 1
        # Constraint: each person has exactly one schedule assignment.
        # TODO: How do we want to handle this sort of filtering? The
        # best option is probably some kind of warning in the API
        # output (without an actual assignment).
        if n_matches > 0:
            solver.Add(solver.Sum(assignments[p_idx][j]
                                  for j in range(n_schedules)) == 1)
    return assignments, costs


def schedule_ordering(schedules_by_cohort):
    """Assigns a canonical ordering to unique schedules across cohorts."""
    schedule_ids = {}
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
            uniq = tuple(uniq)
            if uniq in schedule_ids:
                with_ids.append({'id': schedule_ids[uniq], 'blocks': schedule})
            else:
                schedule_ids[uniq] = sched_id
                schedules_by_id[sched_id] = schedule
                with_ids.append({'id': sched_id, 'blocks': schedule})
                sched_id += 1
        schedules_by_cohort_with_id[cohort] = with_ids
    return schedules_by_id, schedules_by_cohort_with_id


def testing_demand(config, people, n_tests):
    """Calculates testing demand for each person."""
    cohort_counts = defaultdict(int)
    n_people = len(people)
    test_demand = np.zeros(n_people, dtype=np.int)
    for idx, person in enumerate(people):
        cohort_counts[person['cohort']] += 1
        test_demand[idx] = n_tests[person['cohort']]
    return test_demand


def add_schedule_counts(solver, assignments: List[List]):
    """Adds an auxiliary schedule counts vector to the MIP."""
    schedule_counts = []
    n_people = len(assignments)
    n_schedules = len(assignments[0])
    if assignments:
        for i in range(n_schedules):
            sched_count = solver.IntVar(0, n_people, f'count{i}')
            schedule_counts.append(sched_count)
            solver.Add(
                sched_count == solver.Sum(
                    assignments[j][i] for j in range(n_people)
                )
            )
    return schedule_counts



def add_load_balancing(solver, config, people, schedules_by_cohort,
                       schedule_counts, test_demand, start_date, end_date):
    """Adds load balancing constraints (day- and/or block-level) to the MIP."""
    # Determine total testing demand.
    n_people = len(people)
    n_schedules = len(schedule_counts)
    total_demand = test_demand.sum()

    # Generic constraint for site-{block, day} tolerance.
    def site_time_constraint(tol, use_days):
        weights, site_time_ids = site_weights(config, start_date, end_date,
                                               use_days)
        n_site_times = len(site_time_ids)
        site_times = np.zeros((n_schedules, n_site_times))
        for cohort, schedules in schedules_by_cohort.items():
            for schedule in schedules:
                sched_id = schedule['id']
                for block in schedule['blocks']:
                    if use_days:
                        time_hash = (block['date'], block['site'])
                    else:
                        time_hash = (block['start'], block['site'])
                    site_times[sched_id, site_time_ids[time_hash]] = 1

        target_load = total_demand * weights
        min_load = (1 - tol) * target_load
        max_load = (1 + tol) * target_load
        for time_idx in range(n_site_times):
            demand = solver.Sum(site_times[i, time_idx] * schedule_counts[i]
                                for i in range(n_schedules))
            solver.Add(demand >= min_load[time_idx])
            solver.Add(demand <= max_load[time_idx])

    # Constraint: site-blocks are sufficiently load-balanced.
    if 'block_load_tolerance' in config['policy']['bounds']:
        block_load_tol = (config['policy']['bounds']
                          ['block_load_tolerance']['max'])
        site_time_constraint(block_load_tol, use_days=False)

    # Constraint: site-days are sufficiently load-balanced.
    if 'day_load_tolerance' in config['policy']['bounds']:
        day_load_tol = (config['policy']['bounds']
                        ['day_load_tolerance']['max'])
        site_time_constraint(day_load_tol, use_days=True)


def condense_assignments(people: List, schedules: Dict, assignments: List):
    """Converts an assignment matrix to an assignment map."""
    condensed_assignment = {}
    for i, person in enumerate(people):
        condensed_assignment[i] = None
        for j, schedule in enumerate(schedules):
            if (isinstance(assignments[i][j], pywraplp.Variable) and
                    assignments[i][j].solution_value() == 1):
                condensed_assignment[i] = j
                break
    return condensed_assignment


def format_assignments(people: List, schedules: Dict, assignments: Dict):
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

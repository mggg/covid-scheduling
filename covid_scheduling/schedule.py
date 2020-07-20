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


def block_weights(config: Dict,
                  start_date: datetime,
                  end_date: datetime) -> np.ndarray:
    """Determines load-balanced supply for each site-block over a week."""
    n_days = (end_date - start_date).days + 1
    blocks = sorted(config['policy']['blocks'].values(), key=lambda k: k['start'])
    sites = config['sites']
    day_w = np.zeros(len(DAYS) * len(sites))
    block_w = np.zeros(len(DAYS) * len(blocks) * len(sites))
    site_day_ids = {}
    site_day_id = 0
    site_block_ids = {}
    site_block_id = 0
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
                        day_w[site_day_id] += weighted_s
                        block_w[site_block_id] += weighted_s
                start_d = block['start'] - block['start'].replace(**MIDNIGHT)
                site_block_ids[(ts + start_d, site)] = site_block_id
                site_block_id += 1
            site_day_ids[(ts, site)] = site_day_id
            site_day_id += 1
    return {
        'day_weights': day_w / day_w.sum(),
        'block_weights': block_w / block_w.sum(),
        'day_ids': site_day_ids,
        'block_ids': site_block_ids
    }


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
        co: add_sites(
            cohort_schedules(config, co, n_tests[co], blocks),
            config
        )
        for co in config['policy']['cohorts'].keys()
    }

    # Assign an ordering to all unique schedules.
    testing_ids = {}
    testing_schedules = {}
    testing_schedules_by_cohort = {}
    testing_blocks = {}
    testing_sites = {}
    sched_id = 0
    for cohort, schedules in schedules_by_cohort.items():
        with_ids = []
        for schedule in schedules:
            uniq = []
            for block in schedule:
                block_hash = (block['start'], block['end'], block['site'])
                uniq += list(block_hash)
            uniq = tuple(uniq)
            if uniq in testing_ids:
                with_ids.append({'id': testing_ids[uniq], 'blocks': schedule})
            else:
                testing_ids[uniq] = sched_id
                testing_schedules[sched_id] = schedule
                testing_blocks[sched_id] = set((s['date'], s['block'])
                                               for s in schedule)
                testing_sites[sched_id] = set(s['site'] for s in schedule)
                with_ids.append({'id': sched_id, 'blocks': schedule})
                sched_id += 1
        testing_schedules_by_cohort[cohort] = with_ids

    # Cache compatibility sets.
    people_sites = [set(person['site_rank']) for person in people]
    people_blocks = []
    for person in people:
        person_blocks = set()
        for date, blocks in person['schedule'].items():
            for b in blocks:
                person_blocks.add((date, b))
        people_blocks.append(person_blocks)

    # Formulate IP minimum-cost matching problem.
    solver = pywraplp.Solver('SolveAssignmentProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    n_schedules = len(testing_ids)
    costs = np.zeros((n_people, n_schedules))
    assignments = np.zeros((n_people, n_schedules), dtype=np.int).tolist()
    for p_idx, person in tqdm(enumerate(people)):
        cohort = person['cohort']
        n_matches = 0
        for schedule in testing_schedules_by_cohort[cohort]:
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
            if (len(schedule['blocks']) == n_tests[cohort] and
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

    # Add load-balancing constraints (optional).
    if ('day_load_tolerance' in config['policy']['bounds'] or
            'block_load_tolerance' in config['policy']['bounds']):
        # Build the binary site-block and site-day matrices.
        weights = block_weights(config, start_date, end_date)
        n_blocks_per_day = len(config['policy']['blocks'])
        n_sites = len(config['sites'])
        n_site_blocks = len(weights['block_ids'])
        n_site_days = len(weights['day_ids'])
        site_days = np.zeros((n_schedules, n_site_days), dtype=np.int)
        site_blocks = np.zeros((n_schedules, n_site_blocks), dtype=np.int)
        site_day_ids = weights['day_ids']
        site_block_ids = weights['block_ids']
        for cohort, schedules in testing_schedules_by_cohort.items():
            for schedule in schedules:
                sched_id = schedule['id']
                for block in schedule['blocks']:
                    block_hash = (block['start'], block['site'])
                    day_hash = (block['date'], block['site'])
                    site_days[sched_id, site_day_ids[day_hash]] = 1
                    site_blocks[sched_id, site_block_ids[block_hash]] = 1

        # Determine total testing demand.
        cohort_counts = defaultdict(int)
        test_demand = np.zeros(n_people, dtype=np.int)
        for idx, person in enumerate(people):
            cohort_counts[person['cohort']] += 1
            test_demand[idx] = n_tests[person['cohort']]
        total_demand = test_demand.sum()
        print('total demand:', total_demand)

        # Introduce an auxiliary vector to count schedule occurrences.
        schedule_counts = []
        for i in range(n_schedules):
            sched_count = solver.IntVar(0, n_people, f'count{i}')
            schedule_counts.append(sched_count)
            solver.Add(
                sched_count == solver.Sum(
                    assignments[j][i] for j in range(n_people)
                )
            )

        # Constraint: site-blocks are sufficiently load-balanced.
        if 'block_load_tolerance' in config['policy']['bounds']:
            block_load_tol = (config['policy']['bounds']
                              ['block_load_tolerance']['max'])
            target_block_load = total_demand * weights['block_weights']
            min_block_load = (1 - block_load_tol) * target_block_load
            max_block_load = (1 + block_load_tol) * target_block_load
            for sb_idx in range(n_site_blocks):
                sb_demand = solver.Sum(
                    site_blocks[i, sb_idx] * schedule_counts[i]
                    for i in range(n_schedules)
                )
                solver.Add(sb_demand >= min_block_load[sb_idx])
                solver.Add(sb_demand <= max_block_load[sb_idx])

        # Constraint: site-days are sufficiently load-balanced.
        if 'day_load_tolerance' in config['policy']['bounds']:
            day_load_tol = (config['policy']['bounds']
                            ['day_load_tolerance']['max'])
            target_day_load = total_demand * weights['day_weights']
            min_day_load = (1 - day_load_tol) * target_day_load
            max_day_load = (1 + day_load_tol) * target_day_load
            for sd_idx in range(n_site_days):
                sd_demand = solver.Sum(
                    site_days[i, sd_idx] * schedule_counts[i]
                    for i in range(n_schedules)
                )
                solver.Add(sd_demand >= min_day_load[sd_idx])
                solver.Add(sd_demand <= max_day_load[sd_idx])

    # Objective: minimize total matching cost.
    solver.Minimize(solver.Sum([np.dot(a_row, c_row)
                        for (a_row, c_row) in zip(assignments, costs)]))
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        person_assignments = []
        for i, person in enumerate(people):
            assignment = {
                'id': person['id'],
                'campus': person['campus'],
                'cohort': person['cohort'],
                'assigned': False
            }
            for j in range(n_schedules):
                if (isinstance(assignments[i][j], pywraplp.Variable) and
                        assignments[i][j].solution_value() == 1):
                    schedule = testing_schedules[j]
                    schedule_by_date = defaultdict(list)
                    for block in schedule:
                        block_date = block['date'].strftime('%Y-%m-%d')
                        schedule_by_date[block_date].append({
                            'site': block['site'],
                            'block': block['block']
                        })
                    assignment['assigned'] = True
                    assignment['schedule'] = schedule_by_date
                    break
            if not assignment['assigned']:
                assignment['error'] = 'Not enough availability.'
            person_assignments.append(assignment)
        return person_assignments
    if status == pywraplp.Solver.FEASIBLE:
        raise AssignmentError('Could not generate assignments:'
                              'solution is not optimal.')
    # TODO: are there other statuses?
    raise AssignmentError('Could not generate assignments:'
                          'problem is infeasible.')



class AssignmentError(Exception):
    """Raised for errors when generating assignments."""
    def __init__(self, message: str):
        self.message = message


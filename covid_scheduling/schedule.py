"""Campus block scheduling algorithms."""
from typing import Dict, List, Tuple, Callable, Iterable
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations, product
import numpy as np  # type: ignore
from covid_scheduling.costs import schedule_cost
from covid_scheduling.errors import AssignmentError
from covid_scheduling.bipartite import bipartite_assign
from covid_scheduling.constants import SEC_PER_DAY, MIDNIGHT

MAX_DAYS = 14
MAX_TESTS = 2
ASSIGNMENT_METHODS = {'bipartite': bipartite_assign}


def assign_schedules(config: Dict,
                     people: List,
                     start_date: datetime,
                     end_date: datetime,
                     method: str = 'bipartite',
                     cost_fn: Callable = schedule_cost) -> Tuple[List, List]:
    """Assigns people to testing schedules.

    We enumerate the possible testing schedules based on the configuration's
    block schedule, site availability, and testing interval constraints.
    We then solve the person-schedule matching problem using the specified
    method; in particular, we seek an assignment that minimizes the total
    matching cost, where the cost of matching a person with a testing schedule
    is determined by `cost_fn`. This matching problem may also be subject
    to side constraints, such as site-day and site-block load balancing.

    Matches are returned in a schema similar to the person roster's
    schema; any unsuccessful matches (which typically occur due to lack of
    availability) are marked clearly with an error message.

    Args:
        config: A validated university-level configuration.
        people: A validated roster of people.
        start_date: The first day of testing. Only date information is used.
        end_date: The last day of testing (inclusive).
            Only date information is used.
        method: The assignment algorithm to use (only bipartite
            matching is available; heuristic algorithms may be
            added in the future).
        cost_fn: A function which expects a person, a schedule, and a target
            testing interval (in days) and returns a `float` indicating
            the cost of matching the person with the testing schedule.
            In practice, this function should use information about the
            schedule's testing interval, the person's site preferences,
            and the person's testing history. This function does _not_
            need to determine if the person's schedule is compatible
            with the testing schedule; this is handled internally
            by the assignment algorithms.

    Returns:
        Testing schedules for each person (with warnings).

    Raises:
        AssignmentError:
            * When the testing window is too long or starts before it ends.
            * When the specified `method` does not exist.
            * When the assignment problem is invalid with respect to the
              `method` chosen.
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
    for person in people:
        person['schedule'] = {
            date: blocks
            for date, blocks in person['schedule'].items()
            if start_date <= date < (end_date + timedelta(days=1))
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
                                     cost_fn=cost_fn)
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
    """Augments a list of schedules with site permutations.

    For a particular schedule, we can 
    """
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
    """Calculates testing demand for each person based on cohort.

    Args:
        config: The campus-level configuration.
        people: The roster of people to calculate testing demand for.
        n_tests: The number of tests demanded by each cohort for a chosen
            time period. Each cohort should be assigned a positive
            integer number of tests.

    Returns:
        A vector with each person's testing demand.  The testing demand vector
            matches the order of the roster.
    """
    cohort_counts: Dict = defaultdict(int)
    n_people = len(people)
    test_demand = np.zeros(n_people, dtype=np.int)
    for idx, person in enumerate(people):
        cohort_counts[person['cohort']] += 1
        test_demand[idx] = n_tests[person['cohort']]
    return test_demand

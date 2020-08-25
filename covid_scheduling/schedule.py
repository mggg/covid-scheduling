"""Campus block scheduling algorithms."""
from typing import Dict, List, Tuple, Callable, Iterable, Set
from copy import deepcopy
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import combinations, product
import numpy as np  # type: ignore
from covid_scheduling.costs import schedule_cost
from covid_scheduling.errors import AssignmentError
from covid_scheduling.bipartite import bipartite_assign
from covid_scheduling.constants import SEC_PER_DAY, MIDNIGHT
from covid_scheduling.compatibility import (testing_compatibility_sets,
                                            people_compatibility_sets)

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
        config: A validated campus-level configuration.
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
        n_tests = cohort_tests(campus_config, n_days)
        blocks = schedule_blocks(campus_config, start_date, end_date)
        schedules_by_cohort = {
            co: add_sites_to_schedules(
                cohort_schedules(campus_config, co, n_tests[co], blocks),
                campus_config)
            for co in campus_config['policy']['cohorts'].keys()
        }
        schedules, schedules_by_cohort = schedule_ordering(schedules_by_cohort)
        campus_people = [p for p in people if p['campus'] == campus]

        if campus_config['policy']['params'].get('repeat_history', False):
            # Only schedule people who have not been scheduled
            # or are no longer compatible with their last schedule.
            new_people, old_people_with_schedules = repeat_schedules(
                config=campus_config,
                people=campus_people,  #fallback_people,
                start_date=start_date,
                end_date=end_date,
                n_tests=n_tests,
                schedules=schedules,
                schedules_by_cohort=schedules_by_cohort)
        else:
            new_people = campus_people
            old_people_with_schedules = []

        # Assign people to fallback cohorts if necessary.
        if campus_config['policy']['params'].get('fallback_matching', True):
            fallback_people = cohort_fallback(
                config=campus_config,
                people=new_people,
                schedules=schedules,
                schedules_by_cohort=schedules_by_cohort)
        else:
            fallback_people = new_people
        demand = testing_demand(campus_config, fallback_people, n_tests)
        try:
            assign_fn = ASSIGNMENT_METHODS[method]
        except ValueError:
            raise AssignmentError(f'Assignment method "{method}" '
                                  'not available.')

        condensed, stats = assign_fn(config=campus_config,
                                     people=fallback_people,
                                     start_date=start_date,
                                     end_date=end_date,
                                     schedules=schedules,
                                     schedules_by_cohort=schedules_by_cohort,
                                     test_demand=demand,
                                     cost_fn=cost_fn)
        assignments += format_assignments(fallback_people, schedules,
                                          condensed)
        all_stats += stats
    return assignments, all_stats


def format_assignments(people: List, schedules: List,
                       assignments: Dict) -> List[Dict]:
    """Converts an assignment map to the proper JSON schema.

    We mimic the format of a person's input data: the `id`, `cohort`,
    and `campus` fields are retained, and the `schedule` field is replaced with
    the person's assigned schedule instead of all of the person's availability.
    An `assigned` field is added; if `False`, an `error` field is also added.

    Args:
        people: The roster of people to assign.
        schedules: The list of testing schedules people may be assigned to.
        assignments: A dictionary mapping people (by index) to
            either schedules (by index) or `None`.

    Returns:
        A roster of people with testing schedules or errors.
    """
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
            assignment['schedule'] = {}
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


def cohort_fallback(config: Dict, people: List, schedules: List,
                    schedules_by_cohort: Dict) -> List:
    """Moves people into fallback cohorts if their cohorts are too restrictive.

    For instance, a faculty member may fluctuate between testing twice a week
    and once a week depending on their week-to-week schedules. A cohort can
    provide a ranking of fallback cohorts; anyone who fails to meet the
    schedule demands of their original cohort—-that is, they do not have
    enough availability at the proper intervals---is placed in the highest-
    ranked alternate cohort they are compatible with.

    Even with this fallback scheme, a person may not be compatible with
    *any* cohort. This is indicated by replacing the person's `cohort`
    field with `None` rather than a fallback cohort.

    Args:
        config: The campus-level configuration.
        people: The roster of people.
        schedules: The list of testing schedules people may be assigned to.
        schedules_by_cohort: A dictionary with cohort names as the keys and
            lists of schedules as the values.

    Returns:
        A copy of the roster of people. Each person's `cohort` field may
        be retained, replaced with a fallback cohort, or replaced with `None`.
    """
    testing_blocks, testing_sites = testing_compatibility_sets(schedules)
    people_blocks, people_sites = people_compatibility_sets(people)
    people = deepcopy(people)
    for (person, person_blocks, person_sites) in zip(people, people_blocks,
                                                     people_sites):
        primary_cohort = person['cohort']
        cohort_ranking = [primary_cohort]
        cohort_ranking += (config['policy']['cohorts'][primary_cohort].get(
            'fallback', []))
        compatible = False
        for cohort in cohort_ranking:
            for schedule in schedules_by_cohort[cohort]:
                s_idx = schedule['id']
                if (not testing_blocks[s_idx] - person_blocks
                        and not testing_sites[s_idx] - person_sites):
                    compatible = True
                    break
            # As soon as we find a single compatible testing schedule, stop.
            if compatible:
                person['cohort'] = cohort
                break
        if not compatible:
            person['cohort'] = None
    return people


def cohort_tests(config: Dict, n_days: int) -> Dict[str, int]:
    """Determines the number of tests required per person for all cohorts.

    The number of tests required for a person in a cohort is approximately the
    number of testing days divided by the cohort's target testing interval.
    We assume that each person demands at least one test, regardless of their
    cohort's target interval--otherwise, there would be little point in
    requesting a new schedule. We estimate the number of tests for each cohort
    using the cohort's target interval, applying an upper bound of `MAX_TESTS`
    to avoid combinatorial explosion in adversarial cases.

    Args:
        config: The campus-level configuration.
        n_days: The number of days of testing.

    Returns:
        A dictionary mapping each cohort to a positive integer number of tests.
    """
    return {
        name: min(max(int(round(n_days / data['interval']['target'])), 1),
                  MAX_TESTS)
        for name, data in config['policy']['cohorts'].items()
    }


def schedule_blocks(config: Dict, start_date: datetime,
                    end_date: datetime) -> List:
    """Enumerates schedule blocks in a date window.

    Schedule blocks are defined in the `blocks` field of a campus' `policy`
    field. For now, we assume that the block schedule does not change
    from day to day; howerver, the schedules of individual test sites can
    be arbitrarily granular to enable finely regulated load balancing. Blocks
    cannot span multiple days.

    Blocks have the following fields:
        * `date`: Midnight on the day the block is within. (`datetime`)
        * `weekday`: The full name of the weekday the block is within. (`str`)
        * `block`: The label of the block. (`str`)
        * `start`: The start time of the block. (`datetime`)
        * `end`: The end time of the block. (`datetime`)

    Args:
        config: The campus-level configuration.
        start_date: The first day of testing. Only date information is used.
        end_date: The last day of testing (inclusive).
            Only date information is used.

    Returns:
        All blocks in the date window, sorted by start time.
    """
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
                     blocks: List) -> List[Tuple]:
    """Enumerates possible schedules for a cohort.

    A cohort requires `n_tests` per person, usually with a minimum and
    maximum time interval. We enumerate all block combinations of length
    `n_tests` and filter the combinations that fail to meet the constraints.

    Args:
        config: The campus-level configuration (contains testing interval
            constraints).
        cohort: The name of the cohort.
        n_tests: The number of tests required in each schedule.
        blocks: The blocks to build the schedules from (typically generated
            with `schedule_blocks`).

    Returns:
        A list of all valid schedules. Each schedule is a tuple of blocks.
    """
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


def add_sites_to_schedules(schedules: List[Tuple], config: Dict) -> List[List]:
    """Augments a list of schedules with site permutations.

    We consider two primary dimensions when scheduling: *when* a person
    gets tested and *where* a person gets tested. Just as people have
    preferred testing times, they may have preferred testing sites
    depending on the locations of their dormitories and classroom buildings.
    Each test site typically has distinct hours and capacities. Thus,
    it is useful to consider schedules with the same blocks but different
    testing sites separately.

    Given a list of schedules (typically generated with `cohort_schedules`),
    we can make a schedule site-specific in two ways:
        1. For each site, generate a copy of the schedule with *all* blocks
           assigned to the same site. For each schedule, this yields a set of
           augmented schedules with size linear in the number of sites.
        2. For each schedule, enumerate all Cartesian products of sites with
           the length of the schedule---that is, allow the site to vary across
           appointments for each schedule. For each schedule, this yields a set
           of augmented schedules with size exponential in the number of sites.
           This is feasible for a small number of sites.

    By default, we choose the former. Preventing schedules from splitting
    across sites is not just more computationally tractable; the schedules
    this method produces are more user-friendly, as switching between testing
    sites in the same (short) time interval is annoying. However, we retain
    support for the latter. It can be enabled by setting `allow_site_splits` to
    `True` in the configuration's `params` field.

    In either case, we filter out infeasible schedules. If a schedule contains
    a block assigned to a site that is not actually open during that block,
    then the entire schedule is invalid. (If a site is only partially open
    during an assigned block, the schedule is still valid.)

    Args:
        schedules: The schedules to augment.
        config: The campus-level configuration, which includes site data.

    Returns:
        A list of site-augmented schedules.
    """
    site_schedules = []
    sites = config['sites'].keys()
    default_day = {'year': 2020, 'month': 9, 'day': 1}
    allow_splits = config.get('params', {}).get('allow_site_splits', False)
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


def repeat_schedules(config: Dict, people: List, start_date: datetime,
                     end_date: datetime, n_tests: Dict, schedules: List,
                     schedules_by_cohort: Dict) -> Tuple[List, List]:
    """Assigns people to their previous schedule where possible.

    Each person may have a `history` field. If this field is present and
    nonempty, we choose the 

    Args:
        config: The campus-level configuration.
        people: The roster of people to calculate testing demand for.
        start_date: The first day of testing. Only date information is used.
        end_date: The last day of testing (inclusive).
            Only date information is used.
        n_tests: The number of tests demanded by each cohort for a chosen
            time period. Each cohort should be assigned a positive
            integer number of tests.
        schedules: The list of testing schedules people may be assigned to.
        schedules_by_cohort: A dictionary with cohort names as the keys and
            lists of schedules as the values.
    """
    return people, []


def schedule_ordering(schedules_by_cohort: Dict) -> Tuple[List, Dict]:
    """Assigns a canonical ordering to unique schedules across cohorts.

    Multiple cohorts may share the same schedule. In order to minimize the size
    of the assignment problem, we assign an ID to each unique schedule.

    Args:
        schedules_by_cohort: A dictionary with cohort names as the keys and
            lists of schedules as the values.

    Returns:
        A tuple containing:
            * A list of unique schedules, indexed by schedule IDs.
            * A modified version of `schedules_by_cohort`. Each schedule
              is replaced with a wrapper dictionary with an `id` field
              (the unique ID of the schedule) and a `blocks` field
              (the original schedule data).
    """
    schedule_ids: Dict[Tuple, int] = {}
    schedules_by_id = []
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
                schedules_by_id.append(schedule)
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
        if person['cohort'] is not None:
            test_demand[idx] = n_tests[person['cohort']]
    return test_demand

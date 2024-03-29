"""Bipartite matching-based schedule assignment."""
from typing import Dict, List, Tuple, Callable, Union, Any
from datetime import datetime
import numpy as np  # type: ignore
from ortools.linear_solver import pywraplp  # type: ignore
from covid_scheduling.errors import AssignmentError
from covid_scheduling.compatibility import (testing_compatibility_sets,
                                            people_compatibility_sets)
from covid_scheduling.history import history_blocks
from covid_scheduling.load_balancing import site_weights

TIME_LIMIT_MS = 30 * 60 * 1000  # 30-minute time limit for MIP solver


def bipartite_assign(config: Dict, people: List, start_date: datetime,
                     end_date: datetime, schedules: List,
                     schedules_by_cohort: Dict, test_demand: np.ndarray,
                     cost_fn: Callable) -> Tuple[Dict, List]:
    """Assigns people to schedules using bipartite maching.

    We solve a mixed integer program (MIP) that assigns each person to
    at most one schedule such that the total matching cost is minimized.
    (The cost of matching a person with a testing schedule is determined
    by `cost_fn`.)

    This is a _bipartite matching_ problem: people and schedules compose
    the two parts of a bipartite graph, with edges weighted by cost between
    people and schedules when they are compatible.

    Depending on the configuration, the problem may also be subject to
    site-day and site-block load balancing constraints that ensure demand
    does not vary too much at any site over time.

    Args:
        config: The campus-level configuration.
        people: The roster of people to match.
        start_date: The first day of testing. Only date information is used.
        end_date: The last day of testing, inclusive.
            Only date information is used.
        schedules: Schedules by unique ID (starting at 0).
        schedules_by_cohort: Schedules grouped by cohort compatibility;
            some may be duplicates. Each schedule should be in the form
            `{"blocks": <schedule data>, "id": <unique ID>}` for easy
            deduplication. (The schedules in `schedules` do not require
            this wrapper.)
        test_demand: A vector with dimension and indices matching
            `people` indicating the number of tests required by each
            person. Usually, a person's test demand is based entirely on
            their cohort, but we may need to support cases with individual
            exceptions.
        cost_fn: The matching cost function.

    Returns:
        A tuple containing:
            * A dictionary mapping people (by index) to their assigned
              schedule ID if a match is found and `None` otherwise.
            * A list containing person-level matching statistics.

    Raises:
        AssignmentError: If the matching is not optimal or the matching
            problem is infeasible or unbounded. The status code used
            by Google OR-Tools may be included in the error message.
    """
    # Formulate IP minimum-cost matching problem.
    solver = pywraplp.Solver('SolveAssignmentProblem',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    assignments, costs = add_assignments(
        solver=solver,
        config=config,
        people=people,
        schedules=schedules,
        schedules_by_cohort=schedules_by_cohort,
        test_demand=test_demand,
        cost_fn=cost_fn)

    # Objective: minimize total matching cost.
    solver.Minimize(
        solver.Sum([
            np.dot(a_row, c_row) for (a_row, c_row) in zip(assignments, costs)
        ]))

    # Add load-balancing constraints (optional).
    if ('day_load_tolerance' in config['policy']['params']
            or 'block_load_tolerance' in config['policy']['params']):
        # Introduce an auxiliary vector to count schedule occurrences.
        schedule_counts = add_schedule_counts(solver, assignments)
        add_load_balancing(solver=solver,
                           config=config,
                           people=people,
                           schedules_by_cohort=schedules_by_cohort,
                           schedule_counts=schedule_counts,
                           test_demand=test_demand,
                           start_date=start_date,
                           end_date=end_date)

    solver.SetTimeLimit(TIME_LIMIT_MS)
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        condensed = condense_assignments(people, schedules, assignments)
        stats = assignment_stats(people, schedules, test_demand, costs,
                                 condensed)
        return condensed, stats
    if status == pywraplp.Solver.FEASIBLE:
        raise AssignmentError('Could not generate assignments: '
                              'solution is not optimal.')
    raise AssignmentError('Could not generate assignments: problem is '
                          f'infeasible or unbounded (status code {status}).')


def assignment_stats(people: List, schedules: List, test_demand: np.ndarray,
                     costs: np.ndarray, assignment: Dict) -> List:
    """Computes basic statistics about the assignment."""
    stats: List = []
    for p_idx, s_idx in assignment.items():
        if s_idx is None:
            stats.append({})
            continue
        min_cost = np.min(costs[p_idx][costs[p_idx] >= 0])
        actual_cost = costs[p_idx, s_idx]
        site = schedules[s_idx][0]['site']
        # JSON serialization doesn't like NumPy types.
        stats.append({
            'id': people[p_idx]['id'],
            'min_cost': float(min_cost),
            'actual_cost': float(actual_cost),
            'cost_optimal': bool(abs(min_cost - actual_cost) < 1e-10),
            'site_rank': people[p_idx]['site_rank'].index(site),
            'test_demand': int(test_demand[p_idx]),
            'test_supply': len(schedules[s_idx])
        })
    return stats


def add_assignments(solver: pywraplp.Solver, config: Dict, people: List,
                    schedules: List, schedules_by_cohort: Dict,
                    test_demand: np.ndarray,
                    cost_fn: Callable) -> Tuple[np.ndarray, np.ndarray]:
    """Generates assignment and cost matrices.

    The matching problem is constrained by compatibility. A person's schedule
    and a testing schedule are said to be compatible when:
        * The person is available during all blocks in the schedule.
        * The person has ranked all sites in the schedule.
        * The number of blocks in the schedule matches the person's demand.

    Thus, some people may not be matched with a schedule, typically due to
    lack of availability. This is not considered a fatal error; the person
    is simply excluded from the matching.

    When a person and a schedule _are_ compatible, a binary assignment
    variable of type `pywraplp.Solver.IntVar` is added to the problem and
    stored in the returned assignment matrix; the cost of the match is
    stored in the corresponding location in the returned cost matrix. When a
    person has at least one match, a constraint is added to the problem to
    ensure that the sum of assignment variables for that person is exactly
    1---that is, the person is assigned to exactly one schedule.

    When a person and a schedule _are not_ compatible, no assignment variable
    is created; the pairing's entry in the assignment matrix is set to 0,
    and the corresponding cost is set to -1.

    Args:
        solver: The MIP solver to add assignment variables and constraints to.
        config: The campus configuration.
        people: The roster of people to match.
        schedules: Schedules by unique ID (starting at 0).
        schedules_by_cohort: Schedules grouped by cohort compatibility;
            duplicates may exist, so a unique ID is included with each
            schedule.
        test_demand: A vector indicating each person's test demand.
        cost_fn: A function that determines the cost of a matching.

    Returns:
        A tuple containing:
            * The assignment matrix (of mixed type---elements are either
            integers or `pywraplp.Solver.IntVar` objects).
            * The cost matrix corresponding to the assignment matrix.
    """
    testing_blocks, testing_sites = testing_compatibility_sets(schedules)
    people_blocks, people_sites = people_compatibility_sets(people)
    # Generate assignment and cost matrices.
    n_people = len(people)
    n_schedules = len(schedules)
    costs = -1 * np.ones((n_people, n_schedules))
    assignments = np.zeros((n_people, n_schedules), dtype=np.int).tolist()
    for p_idx, person in enumerate(people):
        cohort = person['cohort']
        hist = history_blocks(config, person['history'])
        if hist:
            last_appointment = [hist[-1]]
        else:
            last_appointment = []

        n_matches = 0
        if cohort:
            for schedule in schedules_by_cohort[cohort]:
                s_idx = schedule['id']
                if (len(schedule['blocks']) == test_demand[p_idx]
                        and not testing_blocks[s_idx] - people_blocks[p_idx]
                        and not testing_sites[s_idx] - people_sites[p_idx]):
                    assn = solver.IntVar(0, 1,
                                         f'assignments[{p_idx}, {s_idx}]')
                    assignments[p_idx][s_idx] = assn
                    target = (config['policy']['cohorts'][cohort]['interval']
                              ['target'])
                    costs[p_idx, s_idx] = cost_fn(
                        last_appointment + schedule['blocks'], person, target)
                    n_matches += 1
        # Constraint: each person has exactly one schedule assignment.
        if n_matches > 0:
            solver.Add(
                solver.Sum(assignments[p_idx][j]
                           for j in range(n_schedules)) == 1)
    return np.array(assignments), costs


def add_schedule_counts(
        solver: pywraplp.Solver,
        assignments: np.ndarray) -> List[pywraplp.Solver.IntVar]:
    """Adds an auxiliary schedule counts vector to the MIP.

    The schedule counts vector is used for load balancing; it could
    potentially be useful for other constraints. Constraints are
    added to ensure that each schedule count variable is equal
    to exactly the number of people with a particular schedule.

    Args:
        solver: The MIP solver to add the schedule counts variables
            and constraints to.
        assignments: The assignment matrix used by the MIP solver.

    Returns:
        A list of integer-valued solver variables indicating the number
        of people assigned to a particular schedule, with each index
        corresponding to the schedule's unique ID.
    """
    schedule_counts = []
    n_people = len(assignments)
    n_schedules = len(assignments[0])
    if assignments.size > 0:
        for i in range(n_schedules):
            sched_count = solver.IntVar(0, n_people, f'count{i}')
            schedule_counts.append(sched_count)
            solver.Add(sched_count == solver.Sum(assignments[j, i]
                                                 for j in range(n_people)))
    return schedule_counts


def add_load_balancing(solver: pywraplp.Solver, config: Dict, people: List,
                       schedules_by_cohort: Dict, schedule_counts: List,
                       test_demand: np.ndarray, start_date: datetime,
                       end_date: datetime) -> None:
    """Adds load balancing constraints (day- and/or block-level) to the MIP.

    The `params` field in a campus configuration may contain a site-day load
    balancing constraint in the `site_day_tolerance` field and a site-block
    load balancing cosntraint in the `site_block_tolerance` field.

    Args:
        solver: The MIP solver to add load balancing constraints to.
        campus: The campus-level configuration.
        people: The roster of people to assign.
        schedules_by_cohort: Testing schedules by cohort compatibility.
        schedule_counts: A list of MIP variables indicating the number
            of people assigned to each unique schedule (should be created
            with `add_schedule_counts`).
        test_demand: Number of tests required by each person.
        start_date: The first day of testing. Only date information is used.
        end_date: The last day of testing, inclusive.
            Only date information is used.

    Returns:
        Nothing. The state of `solver` is modified (constraints are added).
    """
    # Determine total testing demand.
    n_people = len(people)
    n_schedules = len(schedule_counts)
    total_demand = test_demand.sum()

    # Generic constraint for site-{block, day} tolerance.
    def site_time_constraint(min_mult, max_mult, use_days):
        weights, site_time_ids = site_weights(config, start_date, end_date,
                                              use_days)
        n_site_times = len(site_time_ids)
        site_times = np.zeros((n_schedules, n_site_times))
        for schedules in schedules_by_cohort.values():
            for schedule in schedules:
                sched_id = schedule['id']
                for block in schedule['blocks']:
                    if use_days:
                        time_hash = (block['date'], block['site'])
                    else:
                        time_hash = (block['start'], block['site'])
                    site_times[sched_id, site_time_ids[time_hash]] = 1

        target_load = total_demand * weights
        min_load = min_mult * target_load
        max_load = max_mult * target_load
        for time_idx in range(n_site_times):
            demand = solver.Sum(site_times[i, time_idx] * schedule_counts[i]
                                for i in range(n_schedules))
            solver.Add(demand >= min_load[time_idx])
            solver.Add(demand <= max_load[time_idx])

    # Constraint: site-blocks are sufficiently load-balanced.
    if 'block_load_tolerance' in config['policy']['params']:
        bound = config['policy']['params']['block_load_tolerance']
        site_time_constraint(bound['min'], bound['max'], use_days=False)

    # Constraint: site-days are sufficiently load-balanced.
    if 'day_load_tolerance' in config['policy']['params']:
        bound = config['policy']['params']['day_load_tolerance']
        site_time_constraint(bound['min'], bound['max'], use_days=True)


def condense_assignments(
        people: List, schedules: List,
        assignments: np.ndarray) -> Dict[int, Union[int, None]]:
    """Converts an assignment matrix to an assignment map.

    The assignment matrix is extremely sparse; there is at most a single 1
    entry in each row. For external purposes, it is best to convert the
    assignment matrix to a dictionary mapping people indices to schedule
    indices. If a person is not assigned to a schedule, their index should
    map to `None` instead.

    Args:
        people: The roster of assigned people.
        schedule: The schedules people are assigned to.
        assignments: The person-testing schedule assignment matrix.

    Returns:
        The person-testing schedule assignment map.
    """
    condensed_assignment: Dict[int, Union[int, None]] = {}
    for i, person in enumerate(people):
        condensed_assignment[i] = None
        for j, schedule in enumerate(schedules):
            if (isinstance(assignments[i, j], pywraplp.Variable)
                    and assignments[i, j].solution_value() == 1):
                condensed_assignment[i] = j
                break
    return condensed_assignment

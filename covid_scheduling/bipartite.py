"""Bipartite matching-based schedule assignment algorithms."""
from typing import Dict, List, Tuple, Callable
from datetime import datetime
import numpy as np
from ortools.linear_solver import pywraplp
from .errors import AssignmentError
from .load_balancing import site_weights

def bipartite_assign(config: Dict,
                     people: List,
                     start_date: datetime,
                     end_date: datetime,
                     schedules: List,
                     schedules_by_cohort: Dict,
                     test_demand: np.ndarray,
                     cost_fn: Callable) -> Dict:
    """Assigns people to schedules using bipartite maching."""
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
        cost_fn=cost_fn
    )

    solver.Minimize(solver.Sum([np.dot(a_row, c_row)
                        for (a_row, c_row) in zip(assignments, costs)]))
    # Add load-balancing constraints (optional).
    if ('day_load_tolerance' in config['policy']['bounds'] or
            'block_load_tolerance' in config['policy']['bounds']):
        # Introduce an auxiliary vector to count schedule occurrences.
        schedule_counts = add_schedule_counts(solver, assignments)
        add_load_balancing(
            solver=solver,
            config=config,
            people=people,
            schedules_by_cohort=schedules_by_cohort,
            schedule_counts=schedule_counts,
            test_demand=test_demand,
            start_date=start_date,
            end_date=end_date
        )

    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        return condense_assignments(people, schedules, assignments)
    if status == pywraplp.Solver.FEASIBLE:
        raise AssignmentError('Could not generate assignments: '
                              'solution is not optimal.')
    raise AssignmentError('Could not generate assignments: problem is '
                          f'infeasible or unbounded (status code {status}).')


def add_assignments(solver: pywraplp.Solver,
                    config: Dict,
                    people: List,
                    schedules: Dict,
                    schedules_by_cohort: Dict,
                    test_demand: np.ndarray,
                    cost_fn: Callable) -> Tuple[List[List], np.ndarray]:
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
    for p_idx, person in enumerate(people):
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
                costs[p_idx, s_idx] = cost_fn(schedule['blocks'], person, target)
                n_matches += 1
        # Constraint: each person has exactly one schedule assignment.
        # TODO: How do we want to handle this sort of filtering? The
        # best option is probably some kind of warning in the API
        # output (without an actual assignment).
        if n_matches > 0:
            solver.Add(solver.Sum(assignments[p_idx][j]
                                  for j in range(n_schedules)) == 1)
    return assignments, costs


def add_schedule_counts(solver: pywraplp.Solver,
                        assignments: List[List]) -> List:
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


def add_load_balancing(solver: pywraplp.Solver,
                       config: Dict,
                       people: List,
                       schedules_by_cohort: Dict,
                       schedule_counts: List,
                       test_demand: np.ndarray,
                       start_date: datetime,
                       end_date: datetime) -> None:
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


def condense_assignments(people: List,
                         schedules: Dict,
                         assignments: List) -> Dict:
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

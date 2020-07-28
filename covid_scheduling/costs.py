"""Matching cost functions."""
from typing import Dict, List
from covid_scheduling.constants import SEC_PER_DAY


def schedule_cost(schedule: List[Dict], person: Dict,
                  target_interval: float) -> float:
    """Computes the cost (based on spacing and site preference) of a matching.

    This is the canonical cost function used for assignments at Tufts; however,
    the assignment algorithms can use alternate cost functions with a matching
    signature.

    We compute the sum of squared deviations of the testing intervals
    from the target interval. Thus, if the schedule contains only
    one appointment, the cost of the schedule is necessarily 0.

    Args:
        schedule: The proposed testing schedule.
        person: The person's data, including their schedule.
        target_interval: The target testing interval, in days.

    Returns:
        The cost of the matching, guaranteed to be non-negative.
    """
    # TODO: Site ranking costs.
    # TODO: Spacing history costs.
    cost = 0
    for left_b, right_b in zip(schedule[:-1], schedule[1:]):
        delta_sec = (right_b['start'] - left_b['start']).total_seconds()
        cost += ((delta_sec / SEC_PER_DAY) - target_interval)**2
    return cost

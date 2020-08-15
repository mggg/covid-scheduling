"""Utility functions for efficient compatibility checks."""
from typing import List, Tuple, Set


def testing_compatibility_sets(schedules: List) -> Tuple[List[Set], List[Set]]:
    """Converts testing schedules to sets for efficient compatibility checks.

    Args:
        schedules: A list of schedules to index.

    Returns:
        A tuple containing:
            * A list mapping each schedule to a set of (date, block) pairs.
            * A list mapping each schedule to a set of sites.
    """
    testing_blocks = [
        set((s['date'], s['block']) for s in sched) for sched in schedules
    ]
    testing_sites = [set(s['site'] for s in sched) for sched in schedules]
    return testing_blocks, testing_sites


def people_compatibility_sets(people: List) -> Tuple[List[Set], List[Set]]:
    """Converts personal schedules to sets for efficient compatibility checks.

    Args:
        people: A list of people to index.

    Returns:
        A tuple containing:
            * A list mapping each person to a set of (date, block) pairs
              corresponding to their availability.
            * A list mapping each person to the set of sites they have ranked.
    """
    people_blocks = []
    for person in people:
        person_blocks = set()
        for date, blocks in person['schedule'].items():
            for b in blocks:
                person_blocks.add((date, b))
        people_blocks.append(person_blocks)
    people_sites = [set(person['site_rank']) for person in people]
    return people_blocks, people_sites

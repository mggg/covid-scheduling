"""Campus block scheduling algorithms."""
import numpy as np
import cvxpy as cp
from datetime import datetime
from typing import Dict, List
from .schemas import DAYS

MAX_DAYS = 14

def site_weights(config: Dict) -> np.ndarray:
    """Determines load-balanced supply for each site-block over a week."""
    blocks = sorted(config['blocks'].values(), key=lambda k: k['start'])
    sites = config['sites']
    weights = np.zeros((len(DAYS) * len(blocks), len(sites)))
    for day_idx, day in enumerate(DAYS):
        for block_idx, block in enumerate(blocks):
            day_block_idx = (day_idx * len(blocks)) + block_idx
            for site_idx, site in enumerate(config['sites']):
                # Determine seconds of overlap between site availability
                # windows and schedule blocks.
                hours = [s for s in site if s['day'] == day]
                for window in hours:
                    start = max(block['start'], window['start'])
                    end = min(block['end'], window['end'])
                    delta_s = (end - start).total_seconds()
                    if delta_s > 0:
                        weighted_s = (delta_s * window['weight'] *
                                      site['n_lines'])
                        weights[day_block_idx, site_idx] += weighted_s
    return weights / weights.sum()


def cohort_schedules(config: Dict, cohort: str):
    pass


def bipartite_assign(config: Dict,
                     people: List,
                     start_date: datetime,
                     end_date: datetime) -> Dict:
    weights = site_weights(config)
    pass


def assign_and_adjust(config: Dict,
                      people: List,
                      start_date: datetime,
                      end_date: datetime) -> Dict:
    pass

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
    window_days = (end_date - start_date).days
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

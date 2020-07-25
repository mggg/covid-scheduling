"""Utility functions for load balancing."""
from datetime import datetime, timedelta
from typing import Dict, Tuple
import numpy as np  # type: ignore
from covid_scheduling.constants import MIDNIGHT


def site_weights(config: Dict,
                 start_date: datetime,
                 end_date: datetime,
                 use_days: bool = False) -> Tuple[np.ndarray, Dict]:
    """Determines the ideal weighting of site-{days, blocks} over an interval.

    Some testing sites may have more capacity than others, and the hours
    of testing sites may vary. We must take this into account when load
    balancing---sites with more capacity should clearly receive a
    proportionately larger share of testing assignments. We compute a
    probability vector over site-blocks (or site-days); testing schedules
    assigned according to these weightings are optimally balanced.

    Args:
        config: The campus-level configuration.
        start_date: The first day of testing. Only date information is used.
        end_date: The last day of testing. Only date information is used.
        use_days: The granularity of the weight vector. If `True`, the
            weight vector will be over site-days (less granular); otherwise,
            it will be over block-days (more granular).

    Returns:
        A tuple containing:
            1. A probability vector representing the ideal weighting of
               site-blocks or site-days. If `use_days` is `True`, the weight
               vector will have (number of days) ⨉ (number of testing sites)
               entries; otherwise, it will have (number of blocks per day) ⨉
               (number of days) ⨉ (number of sites) entries. In both cases,
               the entries will sum to 1 by definition. Entries are ordered
               first by site, then by time unit.
            2. A dictionary mapping (date, site name) or
                (block start time, site name) tuples to probability vector
                indices.
        (TODO: this is a bit clunky---could these be folded together?)
    """
    n_days = (end_date - start_date).days + 1
    blocks = sorted(config['policy']['blocks'].values(),
                    key=lambda k: k['start'])
    sites = config['sites']
    n_units = n_days
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
                hours = [
                    s for s in sites[site]['hours'] if s['day'] == weekday
                ]
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

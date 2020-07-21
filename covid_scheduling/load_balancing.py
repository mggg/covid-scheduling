"""Utility functions for load balancing."""
from datetime import datetime, timedelta
from typing import Dict
import numpy as np
from .constants import DAYS, MIDNIGHT


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

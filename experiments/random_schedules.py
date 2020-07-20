import json
import click
import random
import string
import datetime
import numpy as np
from dateutil.parser import parse as ts_parse
from itertools import permutations
from schedules import ScheduleSampler

n_days = 7
n_blocks = 3
start_date = ts_parse('2020-08-17')
sites = ('51 Winthrop', 'Jackson Gym', 'Curtis Hall')
block_names = ('Morning', 'Afternoon', 'Evening')
campus = 'Medford/Somerville'
site_rankings = {p: 1/6 for p in permutations(sites)}


sampler = ScheduleSampler(
    cohorts={
        'Students': {
            'weight': 5000,
            'schedules': {
                '1111111': 0.7,   # available always
                '1010111': 0.15,  # M/W/F/weekends
                '0101011': 0.15   # Tu/Th/weekends
            },
            'blocks': [1/3, 1/3, 1/3]
        },
        'High-frequency staff': {
            'weight': 500,
            'schedules': {
                '{:07b}'.format(i << 2): 1
                for i in range(2 ** 5) if '{:b}'.format(i).count('1') == 2
            },  # 2 weekdays
            'blocks': [3/4, 1/2, 3/4]
        },
        'Low-frequency staff': {
            'weight': 500,
            'schedules': {
                '{:07b}'.format(i << 2): 1
                for i in range(2 ** 5) if '{:b}'.format(i).count('1') == 1
            },  # 1 weekday
            'blocks': [3/4, 1/2, 3/4]
        }
    },
    n_days=n_days,
    n_blocks=n_blocks
)

@click.command()
@click.option('--out-file', required=True)
@click.option('--n-people', default=5000, type=int)
@click.option('--seed', default=None)
def main(out_file, n_people, seed):
    random.seed(seed)
    np.random.seed(seed)
    sampled_schedules, sampled_cohort = sampler.sample(n_people)
    rows = []
    for schedule, cohort in zip(sampled_schedules, sampled_cohort):
        schedule = [int(s) for s in schedule]
        rank_idx = np.random.choice(range(len(site_rankings)),
                                    p=list(site_rankings.values()))

        day_schedule = {}
        for day in range(n_days):
            day_delta = datetime.timedelta(days=day)
            ts = (start_date + day_delta).strftime('%Y-%m-%d')
            blocks_on = schedule[n_blocks * day:n_blocks * (day + 1)]
            day_blocks = [block_names[idx]
                          for idx, on in enumerate(blocks_on)
                          if on == 1]
            if day_blocks:
                day_schedule[ts] = day_blocks

        rows.append({
            'id': ''.join(random.choice(string.ascii_lowercase +
                                        string.ascii_uppercase)
                          for _ in range(10)),
            'campus': campus,
            'cohort': cohort,
            'schedule': day_schedule,
            'site_rank': list(site_rankings.keys())[rank_idx]
        })

    with open(out_file, 'w') as f:
        json.dump(rows, f)


if __name__ == '__main__':
    main()

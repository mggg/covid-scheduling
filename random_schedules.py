import json
import click
import random
import string
import numpy as np
from schedules import ScheduleSampler

n_days = 7
n_blocks = 3
day_order = ('Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday')
site_rankings = {
    ('Gantcher Center', 'Sophia Gordon Hall'): 0.75,
    ('Sophia Gordon Hall', 'Gantcher Center'): 0.25
}

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
        rows.append({
            'id': ''.join(random.choice(string.ascii_lowercase +
                                        string.ascii_uppercase)
                          for _ in range(10)),
            'cohort': cohort,
            'schedule': {
                day: schedule[n_blocks * idx:n_blocks * (idx + 1)]
                for idx, day in enumerate(day_order)
            },
            'site_rank': list(site_rankings.keys())[rank_idx]
        })
    with open(out_file, 'w') as f:
        json.dump(rows, f)


if __name__ == '__main__':
    main()

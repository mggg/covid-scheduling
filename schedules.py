import numpy as np
from typing import Dict

class ScheduleSampler:
    def __init__(self, cohorts: Dict, n_days: int, n_blocks: int):
        _validate_cohorts(cohorts, n_days, n_blocks)
        self.cohorts = cohorts
        self.n_days = n_days
        self.n_blocks = n_blocks

    def sample(self, n):
        size = self.n_days * self.n_blocks
        sampled_schedules = np.zeros((n, size), dtype=np.int)
        sampled_cohorts = []
        cohort_keys = list(self.cohorts.keys())
        cohort_weights = np.array([self.cohorts[c]['weight']
                                   for c in cohort_keys])
        cohort_probs = cohort_weights / cohort_weights.sum()
        for i in range(n):
            cohort_key = np.random.choice(cohort_keys, p=cohort_probs)
            cohort = self.cohorts[cohort_key]
            # Assemble daily schedule.
            if 'schedules' in cohort:
                schedules = list(cohort['schedules'].keys())
                schedule_weights = np.array([cohort['schedules'][s]
                                             for s in schedules])
                schedule_probs = schedule_weights / schedule_weights.sum()
                random_schedule = np.random.choice(schedules,
                                                   p=schedule_probs)
            else:
                random_schedule = str('1' if np.random.random() < p else '0'
                                      for p in cohort['days'])
            for day, on in enumerate(random_schedule):
                if on == '1':
                    for block in range(self.n_blocks):
                        if np.random.random() < cohort['blocks'][block]:
                            sampled_schedules[i, day * self.n_blocks + block] = 1
            sampled_cohorts.append(cohort_key)
        return sampled_schedules, sampled_cohorts


def _validate_cohorts(cohorts, n_days, n_blocks):
    for name, c in cohorts.items():
        # Population weights
        if 'weight' not in c:
            raise CohortError(f'Cohort "{name}" missing weight')
        if c['weight'] < 0:
            raise CohortError('Cohort "{name}" must have nonnegative weight')

        # Daily schedules
        if 'schedules' in c and not 'days' in c:
            for sched, weight in c['schedules'].items():
                if len(sched) != n_days:
                    raise CohortError('Schedule {sched} in cohort ' +
                                      f'"{name}" must be have length {n_days}')
                if weight < 0:
                    raise CohortError(f'Schedule {sched} in cohort "{name}" ' +
                                      'must have nonnegative weight')
        elif 'days' in c and not 'schedules' in c:
            if len(c['days']) != n_days:
                raise CohortError('Day probability vector of cohort ' +
                                  f'"{name}" must be have length {n_days}')
            if any(p < 0 for p in c['days']):
                raise CohortError('Day probability vector of cohort ' +
                                  f'"{name}" must be nonnegative')
        else:
            raise CohortError(f'Cohort "{name}" must have either schedules ' +
                              'or day probabilities')

        # Block schedules
        if 'blocks' in c:
            if len(c['blocks']) != n_blocks:
                raise CohortError('Block probability vector of cohort ' +
                                  f'"{name}" must be have length {n_blocks}')
            if any(p < 0 for p in c['blocks']):
                raise CohortError('Block probability vector of cohort ' +
                                  f'"{name}" must be nonnegative')


class CohortError(Exception):
    """Raised when a cohort is specified improperly."""


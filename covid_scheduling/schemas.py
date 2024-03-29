"""Schema validation for university configurations and rosters."""
import datetime
from copy import deepcopy
from typing import Dict, List
from dateutil.parser import parse as ts_parse
from schema import Schema, And, Or, Optional, Regex, SchemaError  # type: ignore
from covid_scheduling.constants import DAYS

# YYYY-MM-DD format: https://www.regextester.com/96683
DATE_REGEX = r'([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]))'

# HH:MM:SS format: based on https://regexlib.com/REDetails.aspx?regexp_id=59
HH_MM_SS_REGEX = r'^([0-1][0-9]|[2][0-3]):([0-5][0-9]):([0-5][0-9])$'

# Basic structure is specified declaratively using `Schema` objects.
# When updating the schema, attempt to update these objects first.
# Some complex relationships (such as min/max and start/end relationships,
# as well as uniqueness constraints) cannot be specified declaratively
# in this manner; validation  of these relationships should occur in the
# validation functions.

# STYLE EXCEPTION: Lines in this definiton may be ≤100 characters long.
# yapf: disable
CONFIG_SCHEMA = Schema(
    {
        str: {
            'policy': {
                'blocks': {
                    str: {
                        'start': Regex(
                            HH_MM_SS_REGEX,
                            error='Blocks: start time must be in HH:MM:SS format.'
                        ),
                        'end': Regex(
                            HH_MM_SS_REGEX,
                            error='Blocks: end time must be in HH:MM:SS format.'
                        ),
                    }
                },
                'cohorts': {
                    str: {
                        'interval': {
                            Optional('min'): And(
                                Or(int, float),
                                lambda x: x >= 0,
                                error='Cohorts: minimum interval must be non-negative.'
                            ),
                            Optional('max'): And(
                                Or(int, float),
                                lambda x: x > 0,
                                error='Cohorts: maximum interval must be positive.'
                            ),
                            'target': And(
                                Or(int, float),
                                lambda x: x > 0,
                                error='Cohorts: target interval must be positive.'
                            )
                        },
                        Optional('fallback'): [str]
                    }
                },
                # `bounds` is deprecated; prefer `params`
                Optional(Or('params', 'bounds')): {
                    Optional(Or('day_load_tolerance', 'block_load_tolerance')): {
                        Optional('min'): And(
                            Or(int, float),
                            lambda x: 0 <= x <= 1,
                            error='Bounds: minimum multiplier must be between 0 and 1.'
                        ),
                        'max': And(
                            Or(int, float),
                            lambda x: x >= 1,
                            error='Bounds: maximum multiplier must be at least 1.'
                         )
                    },
                    Optional('allow_site_splits'): bool,
                    Optional('repeat_history'): bool,
                    Optional('fallback_matching'): bool
            }
        },
        'sites': {
            str: {
                'n_lines': And(
                    int,
                    lambda x: x > 0,
                    error='Sites: number of lines must be a positive integer.'
                ),
                'hours': [{
                    'day': And(
                        str,
                        lambda s: s in DAYS,
                        error='Site hours: day must be in ' + str(DAYS)
                    ),
                    'start': Regex(
                        HH_MM_SS_REGEX,
                        error='Site hours: start time must be in HH:MM:SS format.'
                    ),
                    'end': Regex(
                        HH_MM_SS_REGEX,
                        error='Site hours: end time must be in HH:MM:SS format.'
                    ),
                    Optional('weight'): And(
                        Or(int, float),
                        lambda x: x > 0,
                        error='Site hours: weight must be non-negative.'
                    )
                }]
            }
        }
    }
}, ignore_extra_keys=True)

PEOPLE_SCHEMA = Schema([{
    'id': str,
    'campus': str,
    'cohort': str,
    'schedule': Or(
        {Regex(DATE_REGEX): [str]},
        {},
        error='People: schedule keys must be in YYYY-MM-DD format.'
    ),
    Optional('history'): Or(
        {
            Regex(DATE_REGEX): [
                {
                    'site': str,
                    'block': str
                }
            ]
        },
        {},
        error='People: history keys must be in YYYY-MM-DD format.'
    ),
    'site_rank': Or([str], []),
    Optional('last_test'): {
        'date': Regex(DATE_REGEX),
        'block': str
    }
}], ignore_extra_keys=True)
# yapf: enable


def validate_config(config: Dict) -> Dict:
    """"Validates and type-converts a university-level configuration.

    Args:
        config: The raw configuration to be validated.

    Returns:
        The configuration with timestamps converted to `datetime` objects
        and appropriate default fields added.

    Raises:
        SchemaError: If the configuration is malformed.
    """

    config = deepcopy(config)
    CONFIG_SCHEMA.validate(config)
    # Additional checks and conversions:
    #  * For any cohort interval: min <= target <= max
    #  * For any cohort interval: min is 0 if not specified
    #  * For any cohort interval: fallback cohorts are unique and exist.
    #  * For any block: start <= end
    #  * For any site block: start <= end
    #  * For any site block: default weight is 1
    #  * For any site: no overlapping blocks on a given day.
    #  * All times should be `datetime` objects.
    #  * An empty `params` object if one does not already exist.
    #  * Tolerance params have an implicit minimum multiplier of 0
    #    if not otherwise specified.
    for campus in config.values():
        for cohort in campus['policy']['cohorts'].values():
            # TODO: Site ranking costs.
            interval = cohort['interval']
            if 'min' not in interval:
                interval['min'] = 0
            if interval['min'] > interval['target']:
                raise SchemaError('Cohorts: min interval cannot be greater '
                                  'than target interval.')
            if 'max' in interval and interval['max'] < interval['target']:
                raise SchemaError('Cohorts: max interval cannot be less '
                                  'than target interval.')
            if 'fallback' in cohort:
                for fallback in cohort['fallback']:
                    if fallback not in campus['policy']['cohorts']:
                        raise SchemaError('Cohorts: fallback cohort '
                                          f'"{fallback}" does not exist.')
                if len(set(cohort['fallback'])) != len(cohort['fallback']):
                    raise SchemaError('Cohorts: fallback cohorts must '
                                      'be unique.')

        for block in campus['policy']['blocks'].values():
            block['start'] = ts_parse(block['start'])
            block['end'] = ts_parse(block['end'])
            if block['start'] > block['end']:
                raise SchemaError('Blocks: start time cannot be after '
                                  'end time.')

        for site in campus['sites'].values():
            for block in site['hours']:
                block['start'] = ts_parse(block['start'])
                block['end'] = ts_parse(block['end'])
                if block['start'] > block['end']:
                    raise SchemaError('Site hours: start time cannot be after '
                                      'end time.')
                if 'weight' not in block:
                    block['weight'] = 1

            for day in DAYS:
                day_blocks = sorted(
                    [b for b in site['hours'] if b['day'] == day],
                    key=lambda b: b['start'])
                max_time = None
                for block in day_blocks:
                    if max_time and max_time > block['start']:
                        raise SchemaError('Site hours: blocks cannot overlap '
                                          'within a day.')
                    max_time = block['end']

        # Add a dummy params field if necessary.
        # Map `bounds` (deprecated) to `params`.
        campus['policy']['params'] = {
            **campus['policy'].get('bounds', {}),
            **campus['policy'].get('params', {})
        }
        if 'bounds' in campus['policy']:
            del campus['policy']['bounds']

        # Tolerance params have an in implicit minimum of 0 if
        # not otherwise specified.
        for bound in ('day_load_tolerance', 'block_load_tolerance'):
            if bound in campus['policy']['params']:
                bound_min = campus['policy']['params'][bound].get('min', 0)
                campus['policy']['params'][bound]['min'] = bound_min

    return config


def validate_people(people: List, config: Dict) -> List:
    """Validates and type-converts a university-level roster of people.

    Args:
        people: The raw roster to be validated.
        config: The university-level configuration validated by
            `validate_config`.

    Returns:
        The roster with timestamps converted to `datetime` objects.

    Raises:
        SchemaError: If the roster is malformed.
    """
    people = deepcopy(people)
    PEOPLE_SCHEMA.validate(people)
    # Additional checks and conversions:
    # * `campus` must exist in the config.
    # * `cohort` must exist in the config (for `campus`).
    # * All sites in `site_rank` must be unique.
    # * All schedule blocks in `schedule` members must exist in the config.
    # * All schedule blocks in `schedule` members must be unique.
    # * All schedule blocks in `last_test` members must exist in the config.
    # * All dates should be `datetime` objects.
    for person in people:
        campus = person['campus']
        if campus not in config:
            raise SchemaError(f'People: campus "{campus}" does not exist '
                              'in the configuration.')
        campus_config = config[person['campus']]
        cohort = person['cohort']
        if cohort not in campus_config['policy']['cohorts']:
            raise SchemaError(f'People: cohort "{cohort}" does not exist '
                              f'in the configuration for campus "{campus}".')

        sites = campus_config['sites'].keys()
        ranks = person['site_rank']
        if len(set(ranks)) != len(ranks):
            raise SchemaError('People: sites in ranking must be unique.')

        # Sites that do not exist in the configuration are ignored.
        filtered_ranks = [s for s in person['site_rank'] if s in sites]
        person['site_rank'] = filtered_ranks

        # All availabilities must map to existing blocks.
        campus_blocks = campus_config['policy']['blocks'].keys()
        date_schedule = {}
        for date, blocks in person['schedule'].items():
            for block in blocks:
                if block not in campus_blocks:
                    raise SchemaError(f'People: "{block}" does not exist in '
                                      'the block schedule for campus'
                                      f'"{campus}".')
            if len(blocks) != len(set(blocks)):
                raise SchemaError('People: all blocks must be unique for '
                                  'each day in a schedule.')
            date_schedule[ts_parse(date)] = blocks
        person['schedule'] = date_schedule

        # Constraints on history are slightly loose: if an old appointment
        # has a site that no longer exists, it is not filtered out, as it
        # is still useful for determining the appropriate test spacing.
        # However, if invalid blocks are specified, we throw an error---
        # blocks should always be properly mapped by the caller.
        date_history = {}
        history_hashes = []
        for date, appointments in person.get('history', {}).items():
            for appointment in appointments:
                block = appointment['block']
                if block not in campus_blocks:
                    raise SchemaError(f'People: block in history "{block}" '
                                      'does not exist.')
                history_hashes.append(
                    (date, appointment['block'], appointment['site']))
            date_history[ts_parse(date)] = appointments
        if len(history_hashes) != len(set(history_hashes)):
            raise SchemaError('People: appointments in history '
                              'must be unique.')
        person['history'] = date_history

    people_ids = set(p['id'] for p in people)
    if len(people_ids) != len(people):
        raise SchemaError('People: IDs must be unique.')

    return people

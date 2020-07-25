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
CONFIG_SCHEMA = Schema(
    {
        str: {
            'policy': {
                'blocks': {
                    str: {
                        'start':
                        Regex(
                            HH_MM_SS_REGEX,
                            error=
                            'Blocks: start time must be in HH:MM:SS format.'),
                        'end':
                        Regex(
                            HH_MM_SS_REGEX,
                            error='Blocks: end time must be in HH:MM:SS format.'
                        ),
                    }
                },
                'cohorts': {
                    str: {
                        'interval': {
                            Optional('min'):
                            And(Or(int, float),
                                lambda x: x >= 0,
                                error=
                                'Cohorts: minimum interval must be non-negative.'
                                ),
                            Optional('max'):
                            And(Or(int, float),
                                lambda x: x >= 0,
                                error=
                                'Cohorts: maximum interval must be non-negative.'
                                ),
                            'target':
                            And(Or(int, float),
                                lambda x: x >= 0,
                                error=
                                'Cohorts: target interval must be non-negative.'
                                )
                        }
                    }
                },
                Optional('bounds'): {
                    Optional(Or('day_load_tolerance', 'block_load_tolerance')):
                    {
                        'max':
                        And(Or(int, float),
                            lambda x: x >= 0,
                            error='Bounds: maximum tolerance must be positive.'
                            )
                    },
                    Optional('allow_site_splits'): bool
                }
            },
            'sites': {
                str: {
                    'n_lines':
                    And(int,
                        lambda x: x > 0,
                        error=
                        'Sites: number of lines must be a positive integer.'),
                    'hours': [{
                        'day':
                        And(str,
                            lambda s: s in DAYS,
                            error='Site hours: day must be in ' + str(DAYS)),
                        'start':
                        Regex(
                            HH_MM_SS_REGEX,
                            error=
                            'Site hours: start time must be in HH:MM:SS format.'
                        ),
                        'end':
                        Regex(
                            HH_MM_SS_REGEX,
                            error=
                            'Site hours: end time must be in HH:MM:SS format.'
                        ),
                        Optional('weight'):
                        And(Or(int, float),
                            lambda x: x > 0,
                            error='Site hours: weight must be non-negative.')
                    }]
                }
            }
        }
    },
    ignore_extra_keys=True)

PEOPLE_SCHEMA = Schema([{
    'id':
    str,
    'campus':
    str,
    'cohort':
    str,
    'schedule':
    Or({Regex(DATE_REGEX): [str]}, {},
       error='People: schedule keys must be in YYYY-MM-DD format.'),
    'site_rank':
    Or([str], []),
    Optional('last_test'): {
        'date': Regex(DATE_REGEX),
        'block': str
    }
}],
                       ignore_extra_keys=True)


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
    #  * For any block: start <= end
    #  * For any site block: start <= end
    #  * For any site block: default weight is 1
    #  * For any site: no overlapping blocks on a given day.
    #  * All times should be `datetime` objects.
    #  * An empty `bounds` object if one does not already exist.
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

        # Add a dummy bounds field if necessary.
        campus['policy']['bounds'] = campus['policy'].get('bounds', {})
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
    # * All sites in `site_rank` must exist in the config.
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
        for site in ranks:
            if site not in sites:
                raise SchemaError(f'People: site "{site}" does not exist in '
                                  f'the configuration for campus "{campus}".')
        if len(set(ranks)) != len(ranks):
            raise SchemaError('People: sites in ranking must be unique.')

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

        if 'last_test' in person:
            block = person['last_test']['block']
            if block not in campus_blocks:
                raise SchemaError(f'People: "{block}" does not exist in '
                                  'the block schedule for campus'
                                  f'"{campus}".')
            ts = ts_parse(person['last_test']['date'])
            person['last_test']['date'] = ts
        person['schedule'] = date_schedule

    people_ids = set(p['id'] for p in people)
    if len(people_ids) != len(people):
        raise SchemaError('People: IDs must be unique.')

    return people

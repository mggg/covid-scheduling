from copy import deepcopy
from datetime import datetime, timedelta
import pytest
from schema import SchemaError
from covid_scheduling.schemas import validate_config, validate_people

# TODO: The basic declarative validation done by the `Schema` objects
# is not currently tested here--rather, these tests focus on the more
# complicated checks outside of the `Schema` objects. If the schemas
# become significantly more complicated, it may be worthwhile to check
# that they are properly declared.


def test_validate_config_testing_interval_min_target(config_simple_raw):
    config_simple_raw['Campus']['policy']['cohorts']['People']['interval'] = {
        'min': 5,
        'target': 3.5
    }
    with pytest.raises(SchemaError):
        validate_config(config_simple_raw)


def test_validate_config_testing_interval_max_target(config_simple_raw):
    config_simple_raw['Campus']['policy']['cohorts']['People']['interval'] = {
        'min': 1,
        'max': 3,
        'target': 3.5
    }
    with pytest.raises(SchemaError):
        validate_config(config_simple_raw)


def test_validate_config_testing_interval_default_min(config_simple_raw):
    config_simple_raw['Campus']['policy']['cohorts']['People']['interval'] = {
        'target': 3.5
    }
    config = validate_config(config_simple_raw)
    min_interval = (
        config['Campus']['policy']['cohorts']['People']['interval']['min'])
    assert min_interval == 0


def test_validate_config_cohort_fallback_valid(config_simple_raw):
    cohort = config_simple_raw['Campus']['policy']['cohorts']['People'].copy()
    config_simple_raw['Campus']['policy']['cohorts']['People2'] = cohort
    config_simple_raw['Campus']['policy']['cohorts']['People']['fallback'] = [
        'People2'
    ]
    validate_config(config_simple_raw)


def test_validate_config_cohort_fallback_invalid_existence(config_simple_raw):
    config_simple_raw['Campus']['policy']['cohorts']['People']['fallback'] = [
        'People2'
    ]
    with pytest.raises(SchemaError):
        validate_config(config_simple_raw)


def test_validate_config_cohort_fallback_invalid_dupe(config_simple_raw):
    cohort = config_simple_raw['Campus']['policy']['cohorts']['People'].copy()
    config_simple_raw['Campus']['policy']['cohorts']['People2'] = cohort
    config_simple_raw['Campus']['policy']['cohorts']['People']['fallback'] = [
        'People2', 'People2'
    ]
    with pytest.raises(SchemaError):
        validate_config(config_simple_raw)


def test_validate_config_schedule_block_start_end_order(config_simple_raw):
    start = config_simple_raw['Campus']['policy']['blocks']['Block']['start']
    end = config_simple_raw['Campus']['policy']['blocks']['Block']['end']
    config_simple_raw['Campus']['policy']['blocks']['Block']['start'] = end
    config_simple_raw['Campus']['policy']['blocks']['Block']['end'] = start
    with pytest.raises(SchemaError):
        validate_config(config_simple_raw)


def test_validate_config_site_block_start_end_order(config_simple_raw):
    for block in config_simple_raw['Campus']['sites']['Testing']['hours']:
        start = block['start']
        end = block['end']
        block['start'] = end
        block['end'] = start
    with pytest.raises(SchemaError):
        validate_config(config_simple_raw)


def test_validate_config_site_block_default_weight(config_simple):
    for block in config_simple['sites']['Testing']['hours']:
        assert block['weight'] == 1


def test_validate_config_site_block_start_no_overlap(config_simple_raw):
    site = config_simple_raw['Campus']['sites']['Testing']
    day = site['hours'][0]['day']
    start = site['hours'][0]['start']
    new_blocks = [{
        'day': day,
        'start': start,
        'end': start.replace('08', '12')
    }, {
        'day': day,
        'start': start.replace('08', '10'),
        'end': start.replace('08', '14')
    }]
    site['hours'] = new_blocks + site['hours'][1:]
    with pytest.raises(SchemaError):
        validate_config(config_simple_raw)


def test_validate_config_datetime(config_simple):
    for block in config_simple['policy']['blocks'].values():
        assert isinstance(block['start'], datetime)
        assert isinstance(block['end'], datetime)
    for block in config_simple['sites']['Testing']['hours']:
        assert isinstance(block['start'], datetime)
        assert isinstance(block['end'], datetime)


def test_validate_config_params(config_simple):
    assert config_simple['policy']['params'] == {}


def test_validate_config_params_min_mult(config_simple_raw):
    params = {
        'day_load_tolerance': {
            'max': 1.25
        },
        'block_load_tolerance': {
            'max': 1.25
        }
    }
    config_simple_raw['Campus']['policy']['params'] = params
    config = validate_config(config_simple_raw)
    for key in params:
        assert config['Campus']['policy']['params'][key]['min'] == 0


def test_validate_config_bounds_mapped_to_params(config_simple_raw):
    config_simple_raw['Campus']['policy']['params'] = {
        'day_load_tolerance': {
            'max': 1.25
        }
    }
    config_simple_raw['Campus']['policy']['bounds'] = {
        'block_load_tolerance': {
            'max': 1.25
        }
    }
    config = validate_config(config_simple_raw)
    for k in ('block_load_tolerance', 'day_load_tolerance'):
        assert k in config['Campus']['policy']['params']
    assert 'bounds' not in config['Campus']['policy']


def test_valiaate_people_baseline(people_simple_raw, config_simple_all):
    assert validate_people(people_simple_raw, config_simple_all)


def test_validate_people_nonexistent_campus(people_simple_raw,
                                            config_simple_all):
    people_simple_raw[0]['campus'] = 'bad'
    with pytest.raises(SchemaError):
        validate_people(people_simple_raw, config_simple_all)


def test_validate_people_nonexistent_cohort(people_simple_raw,
                                            config_simple_all):
    people_simple_raw[0]['cohort'] = 'bad'
    with pytest.raises(SchemaError):
        validate_people(people_simple_raw, config_simple_all)


def test_validate_people_site_rank_nonexistent_site(people_simple_raw,
                                                    config_simple_all):
    people_simple_raw[0]['site_rank'] = ['bad']
    with pytest.raises(SchemaError):
        validate_people(people_simple_raw, config_simple_all)


def test_validate_people_site_rank_duplicate_site(people_simple_raw,
                                                  config_simple_all):
    people_simple_raw[0]['site_rank'] = ['Testing', 'Testing']
    with pytest.raises(SchemaError):
        validate_people(people_simple_raw, config_simple_all)


def test_validate_people_schedule_nonexistent_site(people_simple_raw,
                                                   config_simple_all):
    people_simple_raw[0]['schedule']['2020-01-01'] = ['bad']
    with pytest.raises(SchemaError):
        validate_people(people_simple_raw, config_simple_all)


def test_validate_people_schedule_duplicate_site(people_simple_raw,
                                                 config_simple_all):
    people_simple_raw[0]['schedule']['2020-01-01'] = ['Block', 'Block']
    with pytest.raises(SchemaError):
        validate_people(people_simple_raw, config_simple_all)


def test_validate_people_schedule_datetime(people_simple, config_simple_all):
    for key in people_simple[0]['schedule']:
        assert isinstance(key, datetime)


def test_validate_people_unique_ids(people_simple_raw, config_simple_all):
    people_simple_raw.append(deepcopy(people_simple_raw[0]))
    with pytest.raises(SchemaError):
        validate_people(people_simple_raw, config_simple_all)

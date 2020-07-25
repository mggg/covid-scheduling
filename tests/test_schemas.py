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


def test_validate_config_bounds(config_simple):
    assert config_simple['policy']['bounds'] == {}


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

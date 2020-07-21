import json
from dateutil.parser import parse as ts_parse
from covid_scheduling import (
    validate_people, validate_config, assign_schedules
)

if __name__ == '__main__':
    with open('data/tufts.json') as f:
        config = validate_config(json.load(f)['campuses'])
    with open('data/tufts_sample_5000.json') as f:
        people = validate_people(json.load(f), config)
    print(assign_schedules(config,
                           people[:100],
                           ts_parse('2020-08-17'),
                           ts_parse('2020-08-23')))

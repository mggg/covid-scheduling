import json
from dateutil.parser import parse as ts_parse
from covid_scheduling import (
    validate_people, validate_config, bipartite_assign
)

if __name__ == '__main__':
    with open('../../tufts.json') as f:
        config = validate_config(json.load(f)['campuses'])
    with open('../../tufts_sample_5000.json') as f:
        people = validate_people(json.load(f), config)
    print(bipartite_assign(config['Medford/Somerville'],
                           people[:300],
                           ts_parse('2020-08-17'),
                           ts_parse('2020-08-23')))

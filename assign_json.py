import json
import click
import random
import numpy as np
from dateutil.parser import parse as ts_parse
from covid_scheduling import (
    validate_config, validate_people, bipartite_assign, AssignmentError
)


@click.command()
@click.option('--config-file', required=True)
@click.option('--people-file', required=True)
@click.option('--out-file', required=True)
@click.option('--start-date', required=True)
@click.option('--end-date', required=True)
@click.option('--seed', default=None)
def main(config_file, people_file, out_file, start_date, end_date, seed):
    random.seed(seed)
    np.random.seed(seed)
    start_ts = ts_parse(start_date)
    end_ts = ts_parse(end_date)

    with open(config_file) as f:
        config_raw = json.load(f)
    with open(people_file) as f:
        people_raw = json.load(f)
    config = validate_config(config_raw['campuses'])
    people = validate_people(people_raw, config)

    assignments = []
    for campus, campus_config in config.items():
        campus_people = [p for p in people if p['campus'] == campus]
        assignments += bipartite_assign(campus_config,
                                        campus_people,
                                        start_ts,
                                        end_ts)
    with open(out_file, 'w') as f:
        json.dump({'people': assignments}, f)


if __name__ == '__main__':
    main()

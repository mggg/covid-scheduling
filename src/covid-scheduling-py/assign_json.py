import json
import click
import random
import numpy as np
from dateutil.parser import parse as ts_parse
from covid_scheduling import validate_campus, ConfigError
from pprint import pprint


@click.command()
@click.option('--config-file', required=True)
@click.option('--students-file', required=True)
@click.option('--out-file', required=True)
@click.option('--campus', required=True)
@click.option('--start-date', required=True)
@click.option('--end-date', required=True)
@click.option('--seed', default=None)
def main(config_file, students_file, out_file, campus,
         start_date, end_date, seed):
    random.seed(seed)
    np.random.seed(seed)
    start_ts = ts_parse(start_date)
    end_ts = ts_parse(end_date)

    try:
        with open(config_file) as f:
            config = json.load(f)
    except IOError:
        raise ConfigError(f'Configuration file "{config_file}" not found.')
    campus_config = validate_campus(campus, config['campuses'].get(campus, {}))
    pprint(campus_config)


if __name__ == '__main__':
    main()


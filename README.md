# COVID-19 Test Scheduling

This repository contains scheduling algorithms to assign people (students, faculty, and staff) at universities to surveillance testing schedules. Surveillance testing is a key component of many campus reopening strategies, and the logistics of testing thousands of busy people multiple times a week without long lines at testing centers are nontrivial. Tufts University is developing a campus monitoring app that allows members of the testing population to specify their testing availability and preferred testing locations. This tool aggregates personal schedules and university-level information (such as the desired testing frequencies for population cohorts) to produce a scheduling with optimal schedules that satisfy the population's preferences and epidemiological needs as well as possible while avoiding testing center overload.

## Getting started
This tool has been most extensively tested with Python 3.8, but it should work with Python 3.6 or above. For development, we recommend installing all dependencies in a virtual environment managed with [Anaconda](https://www.anaconda.com/). All dependencies can be installed with `pip install -r requirements.txt`. To deploy to Heroku, [create an app](https://devcenter.heroku.com/articles/creating-apps) and push using `git push heroku main`.

## Inputs
### University-level configuration
An example university configuration, loosely based on Tufts' reopening plans, is available at `data/tufts.json`. The `policy` section contains parameters that are not likely to change frequently, such as the testing block schedule (as displayed in the app), cohort information, and load balancing tolerances. The `sites` section contains the hours and capacity of individual testing sites; these hours are likely to change from week to week.

### Personal availability
An example roster of 5,000 randomly generated personal schedules is available at `data/tufts_sample_5000.json`. Use the `experiments/random_schedules.py` script to generate more random schedules; the `ScheduleSampler` in this script can be adjusted to vary the relative weighting of cohorts and the allowable schedules within each cohort.

## Command-line interface
To generate assignments from the command line, run

```
python assign_json.py --config-file <configuration> --people-file <roster of people> --out-file <output file> --start-date <start date> --end-date <end date, inclusive>
```

For example, to sample schedules for the week of Monday, August 17, 2020 using the example configuration and people roster, run
```
python assign_json.py --config-file data/tufts.json --people-file data/tufts_sample_5000.json --out-file assign.json --start-date 2020-08-17 --end-date 2020-08-23
```

**Note: the scheduler currently supports only weeklong intervals starting on a Monday. This is a known limitation and will be resolved shortly.**

## API
To start a simple Flask-based JSON API is available; execute `FLASK_DEBUG=1 flask run` to start the development server. The API accepts `POST` requests to the root endpoint of the format
```json
{
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "config": { <configuration> },
  "people": { <roster of people> }
}
```

The output schema of a successful request is
```json
{
  "people": { <roster of people, with assignments> },
  "stats": { <person-level assignment statistics> }
}
```

## The algorithm
`covid_scheduling/bipartite.py` contains an implementation of a mixed-integer program for bipartite matching with optional load-balancing constraints. People are matched to schedules such that total matching cost is minimized; the cost of a person/testing schedule match is currently determined solely by the spacing of the testing schedule (schedules that closely match the person's desired testing interval are cheap, while improperly spaced schedules are expensive); we intend to augment the cost function to consider testing history and site preferences.

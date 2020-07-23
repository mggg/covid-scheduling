"""Minimal POST-based API for test scheduling."""
from datetime import datetime
from flask import Flask, jsonify, request
from schema import SchemaError
from covid_scheduling import (validate_people, validate_config,
                              assign_schedules, AssignmentError)

app = Flask(__name__)


@app.route('/', methods=['POST'])
def schedule():
    """Main endpoint for schedule assignment."""
    if not request.is_json:
        raise InvalidUsage('Expected content-type is application/json.')
    body = request.get_json()
    if 'config' not in body:
        raise InvalidUsage("Expected campus configurations in 'config' field.")
    if 'people' not in body:
        raise InvalidUsage("Expected a 'people' field.")

    try:
        start_date = datetime.strptime(body.get('start', ''), '%Y-%m-%d')
    except (TypeError, ValueError):
        raise InvalidUsage("Expected a start date in field 'start' "
                           "with format YYYY-MM-DD.")
    try:
        end_date = datetime.strptime(body.get('end', ''), '%Y-%m-%d')
    except (TypeError, ValueError):
        raise InvalidUsage("Expected a start date in field 'end' "
                           "with format YYYY-MM-DD.")

    try:
        config = validate_config(body['config'])
    except SchemaError as ex:
        raise InvalidUsage('Could not validate configuration.',
                           payload={'fields': ex.autos})
    try:
        people = validate_people(body['people'], config)
    except SchemaError as ex:
        raise InvalidUsage('Could not validate people.',
                           payload={'fields': ex.autos})

    try:
        assignments, stats = assign_schedules(config, people, start_date,
                                              end_date)
    except AssignmentError as ex:
        raise InvalidUsage(f'Assignment error: {ex.message}', 500)
    except Exception as ex:
        raise InvalidUsage('Unknown assignment error.', 500)
    return jsonify({'people': assignments, 'stats': stats})


# Error handling template from
# https://flask.palletsprojects.com/en/1.1.x/patterns/apierrors/
class InvalidUsage(Exception):
    """Raised for assignment failures."""
    def __init__(self, message, status_code=400, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        """Serializes the error."""
        rv = dict(self.payload or ())
        rv['error'] = self.message
        return rv


@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    """Handler for assignment failures."""
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response


if __name__ == '__main__':
    app.run(threaded=True, port=5000)

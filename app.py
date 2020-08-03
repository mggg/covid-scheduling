"""Minimal POST-based API for test scheduling."""
import os
from datetime import datetime
from flask import Flask, jsonify, request
from schema import SchemaError
from covid_scheduling import (validate_people, validate_config,
                              assign_schedules, AssignmentError)
from celery import Celery
from celery.utils.log import get_task_logger


app = Flask(__name__)


# Celery configuration template from
# https://flask.palletsprojects.com/en/1.1.x/patterns/celery/
def make_celery(app):
    """Initializes Celery with Flask context."""
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL'],
        task_track_started=app.config['CELERY_TRACK_STARTED']
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

app.config.update(
    CELERY_BROKER_URL=os.getenv('REDIS_URL', ''),
    CELERY_RESULT_BACKEND=os.getenv('REDIS_URL', ''),
    CELERY_TRACK_STARTED=True
)
celery = make_celery(app)
task_logger = get_task_logger(__name__)


@celery.task(bind=True)
def run_scheduler(self, body):
    params = parse_request(body)
    try:
        assignments, stats = assign_schedules(
            params['config'],
            params['people'],
            params['start_date'],
            params['end_date']
        )
    except AssignmentError as ex:
        task_logger.error('Assignment error.', exc_info=True)
        self.update_state(state='FAILURE', meta={
            'error': f'Assignment error: {ex.message}'
        })
    except Exception as ex:
        task_logger.error('Unknown assignment error.', exc_info=True)
        self.update_state(state='FAILURE', meta={
            'error': 'Unknown assignment error.'
        })
    return {'people': assignments, 'stats': stats}


@app.route('/', methods=['POST'])
def schedule():
    """Main endpoint for synchronous schedule assignment."""
    if not request.is_json:
        raise InvalidUsage('Expected content-type is application/json.')
    body = request.get_json()
    params = parse_request(body)

    try:
        assignments, stats = assign_schedules(
            params['config'],
            params['people'],
            params['start_date'],
            params['end_date']
        )
    except AssignmentError as ex:
        app.logger.error('Assignment error.', exc_info=True)
        raise InvalidUsage(f'Assignment error: {ex.message}', 500)
    except Exception as ex:
        app.logger.error('Unknown assignment error.', exc_info=True)
        raise InvalidUsage('Unknown assignment error.', 500)
    return jsonify({'people': assignments, 'stats': stats})


@app.route('/jobs', methods=['POST'])
def start_job():
    """Main endpoint for starting asynchronous jobs."""
    if not request.is_json:
        raise InvalidUsage('Expected content-type is application/json.')
    body = request.get_json()
    parse_request(body)  # Validate configuration before starting job.
    task = run_scheduler.delay(body)
    return jsonify({'status': 'pending', 'id': task.id})


@app.route('/jobs/<task_id>', methods=['GET'])
def job_status(task_id):
    task = run_scheduler.AsyncResult(task_id)
    if task.state == 'FAILURE':
        response = {
            'status': 'failed',
            'error': task.info.get('error', 'Unknown error.')
        }
    elif task.state == 'SUCCESS':
        response = {'status': 'succeeded', 'result': task.info}
    else:
        response = {'status': task.state.lower()}
    return jsonify(response)


def parse_request(body):
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
    return {
        'config': config,
        'people': people,
        'start_date': start_date,
        'end_date': end_date
    }


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

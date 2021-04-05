"""Helper functions for handling appointment history."""
from typing import List, Dict
from covid_scheduling.constants import MIDNIGHT


def history_blocks(config: Dict, history: Dict) -> List[Dict]:
    """Maps an appointment history (block label format) to start/end time format."""
    new_appointments: List[Dict] = []
    for ts, appointments in history.items():
        for appointment in appointments:
            block = config['policy']['blocks'][appointment['block']]
            start_delta = block['start'] - block['start'].replace(**MIDNIGHT)
            end_delta = block['end'] - block['end'].replace(**MIDNIGHT)
            new_appointments.append({
                'date': ts,
                'weekday': ts.strftime('%A'),
                'block': appointment['block'],
                'site': appointment['site'],
                'start': ts + start_delta,
                'end': ts + end_delta
            })
    return sorted(new_appointments, key=lambda b: b['start'])

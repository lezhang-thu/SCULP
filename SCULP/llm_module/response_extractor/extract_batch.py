import re
from typing import List, Any

from .extract_base import Extract


class BatchExtract(Extract):

    @staticmethod
    def extract(raw_response, num_max):
        pattern = r'LogTemplate.*?`([^`]+)`'
        matches = re.findall(pattern, raw_response)
        return [{
            'idx': idx + 1,
            'template': tmpl
        } for idx, tmpl in enumerate(matches[-num_max:])]

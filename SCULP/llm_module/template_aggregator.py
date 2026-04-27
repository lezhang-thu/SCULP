from collections import Counter
from ..utils import validate_template


def aggregate_by_majority(logs, templates=[]):
    templates = [t for t in templates if validate_template(t)]
    if len(templates) == 0:
        return ""
    else:
        counter = Counter(templates)
        mode_template = counter.most_common(1)[0][0]
        return mode_template

import regex as re


def match_template(log: str, template: str) -> bool:
    """
    Check whether a log matches a template.
    In the template, '<*>' can match any string (including empty).
    """
    # Escape regex special chars except the <*> pattern
    # Step 1: temporarily replace <*> with a placeholder
    placeholder = "__WILDCARD__"
    temp = template.replace("<*>", placeholder)

    # Step 2: escape everything else for regex safety
    temp = re.escape(temp)

    # Step 3: replace placeholder back with regex '.*'
    regex_pattern = "^" + temp.replace(placeholder, ".*") + "$"

    # Step 4: match
    return re.match(regex_pattern, log) is not None


def verify_template_for_log_with_first_token(log, template):
    """ Not always True, just test speed """
    tok_log, tok_temp = log.split(), template.split()
    if "<*>" not in tok_temp[0] and tok_log[0] != tok_temp[0]:
        return False
    if "<*>" not in tok_temp[-1] and tok_log[-1] != tok_temp[-1]:
        return False
    # debug
    #print('so far so good')
    #return verify_template_for_log_regex(log, template)
    return match_template(log, template)


def validate_template(template):
    if len(template) == 0:
        return False
    if template.count("<*>") > 50:
        return False

    return True


def preprocess_log_for_query(log, regexes):
    for currentRex in regexes:
        log = re.sub(currentRex, "<*>", log)
    return log

import re
import string
import regex as re

param_regex = [r'{([ :_#.\-\w\d]+)}', r'{}']


def process_string(s):
    # Tokenize by whitespace
    tokens = s.split()

    # Process each token
    processed_tokens = []
    for token in tokens:
        # Keep track of where we've processed up to
        search_start = 0

        # Process each '<*>' sequentially
        while True:
            # Find next '<*>' starting from search_start
            marker = '<*>'
            pos = token.find(marker, search_start)

            if pos == -1:
                # No more '<*>' found
                break

            # Get the part before '<*>'
            before = token[:pos]

            # Get the part after '<*>'
            after = token[pos + 3:]  # Skip '<*>' (3 characters)

            end_chars = ('(', ')', '[', ']', '{', '}', ',', '.', ':', '<', '>',
                         '#', '$', '/', '"', "'")
            first_end_pos = -1

            for i, char in enumerate(after):
                if char in end_chars:
                    first_end_pos = i
                    break

            # Build the processed token
            if first_end_pos >= 0:
                # Found an end character, keep it and everything after
                token = before + marker + after[first_end_pos:]
            else:
                # No end character found, remove everything after '<*>'
                token = before + marker

            # Move search_start past the current '<*>' we just processed
            search_start = pos + 3

        processed_tokens.append(token)

    # Concatenate tokens back with spaces
    return ' '.join(processed_tokens)


def correct_single_template(template, user_strings=None):
    """Apply all rules to process a template.

    DS (Double Space)
    BL (Boolean) # we don't use this
    US (User String) # we don't use this
    DG (Digit)
    HEX (Hex Variables)
    PS (Path-like String) # we don't use this
    WV (Word concatenated with Variable)
    DV (Dot-separated Variables)
    CV (Consecutive Variables)

    """

    # boolean = {}
    # default_strings = {}
    path_delimiters = {  # reduced set of delimiters for tokenizing for checking the path-like strings
        r'\s', r'\,', r'\!', r'\;', r'\:', r'\=', r'\|', r'\"', r'\'', r'\[',
        r'\]', r'\(', r'\)', r'\{', r'\}'
    }
    token_delimiters = path_delimiters.union({  # all delimiters for tokenizing the remaining rules
        r'\.', r'\-', r'\+', r'\@', r'\#', r'\$', r'\%', r'\&',
    })

    # apply DS
    template = template.strip()
    template = re.sub(r'\s+', ' ', template)

    # tokenize for the remaining rules
    tokens = re.split('(' + '|'.join(token_delimiters) + ')',
                      template)  # tokenizing while keeping delimiters
    new_tokens = []
    for token in tokens:
        # apply DG
        if re.match(r'^\d+$', token):
            token = '<*>'

        # apply Hex
        if re.match(r'0x[0-9a-fA-F]+', token):
            token = '<*>'
        while "0x<*>" in token:
            token = token.replace("0x<*>", "<*>")

        # apply WV
        if re.match(r'^[^\s\/]+<\*>[^\s\/]+$', token):
            if token != '<*>/<*>':  # need to check this because `/` is not a deliminator
                token = '<*>'

        # collect the result
        new_tokens.append(token)
    # make the template using new_tokens
    template = ''.join(new_tokens)
    template = process_string(template)

    # Substitute consecutive variables only if separated with any delimiter including "." (DV)
    while True:
        prev = template
        template = re.sub(r'<\*>\.<\*>', '<*>', template)
        if prev == template:
            break

    # Substitute consecutive variables only if not separated with any delimiter including space (CV)
    # NOTE: this should be done at the end
    while True:
        prev = template
        template = re.sub(r'<\*><\*>', '<*>', template)
        if prev == template:
            break

    while " #<*># " in template:
        template = template.replace(" #<*># ", " <*> ")

    while " #<*> " in template:
        template = template.replace(" #<*> ", " <*> ")

    while "<*>:<*>" in template:
        template = template.replace("<*>:<*>", "<*>")

    while "<*>#<*>" in template:
        template = template.replace("<*>#<*>", "<*>")

    while "<*>/<*>" in template:
        template = template.replace("<*>/<*>", "<*>")

    while "<*>@<*>" in template:
        template = template.replace("<*>@<*>", "<*>")

    while "<*>.<*>" in template:
        template = template.replace("<*>.<*>", "<*>")

    while '"<*>"' in template:
        template = template.replace('"<*>"', '<*>')
    while "'<*>'" in template:
        template = template.replace("'<*>'", "<*>")

    while "<*><*>" in template:
        template = template.replace("<*><*>", "<*>")

    while "( <*>, <*>)" in template:
        template = template.replace("( <*>, <*>)", "(<*>, <*>)")
    template = template.replace('`', '')

    while " <*>. " in template:
        template = template.replace(" <*>. ", " <*> ")
    while " <*>, " in template:
        template = template.replace(" <*>, ", " <*> ")

    while "<*>+<*>" in template:
        template = template.replace("<*>+<*>", "<*>")
    while "<*>##<*>" in template:
        template = template.replace("<*>##<*>", "<*>")
    while "#<*>#" in template:
        template = template.replace("#<*>#", "<*>")
    while "<*>-<*>" in template:
        template = template.replace("<*>-<*>", "<*>")
    while " <*> <*> " in template:
        template = template.replace(" <*> <*> ", " <*> ")
    while template.endswith(" <*> <*>"):
        template = template[:-8] + " <*>"
    while template.startswith("<*> <*> "):
        template = "<*> " + template[8:]

    while "<*>,<*>" in template:
        template = template.replace("<*>,<*>", "<*>")
    while "(<*> <*>)" in template:
        template = template.replace("(<*> <*>)", "(<*>)")
    while " /<*> " in template:
        template = template.replace(" /<*> ", " <*> ")
    if template.endswith(" /<*>"):
        template = template[:-5] + " <*>"

    # Attribute key-value pair
    if template.count("=<*>") >= 3:
        template = template.replace("= ", "=<*> ")
    return template


def post_process_template(template, regs_common):
    template = re.sub(r'\{[A-Za-z0-9_-]+\}', '<*>', template)
    template = correct_single_template(template)
    static_part = template.replace("<*>", "")
    punc = string.punctuation
    for s in static_part:
        if s != ' ' and s not in punc:
            print(f"\tPost Template: `{template}`")
            return template, True
    print("Get a too general template. Error.")
    return "", False


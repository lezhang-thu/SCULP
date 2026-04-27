import regex as re
import numpy as np
from SCULP.llm_module.post_process import post_process_template
from SCULP.utils import validate_template, verify_template_for_log_with_first_token

alphabet_set = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
smallletter_set = set("abcdefghijklmnopqrstuvwxyz")
bigletter_set = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def split_template(template, split=[" "]):
    """
    Split a log template into parts based on the specified split characters.

    :param template: The log template to be split.
    :param split: A list of characters to split the template by. Defaults to [" "].
    :return: The split parts of the template.
    """
    pattern = '|'.join(re.escape(s) for s in split)
    result = re.split(pattern, template)
    result = [part for part in result if part]
    return result


def split_template_naive(template):
    """
    Split a log template into parts using the default space character.

    :param template: The log template to be split.
    :return: A list of parts obtained by splitting the template.
    """
    return template.split(" ")


def jaccard_similarity(parts1, parts2):
    """
    Calculate the Jaccard similarity between two sets of template parts.

    :param parts1: The first set of template parts.
    :param parts2: The second set of template parts.
    :return: The Jaccard similarity score.
    """
    common = set(parts1).intersection(parts2)
    union = set(parts1).union(parts2)
    return (len(common) + 0.00001) / (len(union) + 0.00001)


def merge_template_by_star(template1, template2, split=[" "]):
    """
    Merge two templates using the '*' placeholder.

    :param template1: The new template.
    :param template2: The old template in the template database.
    :param split: A list of characters to split the templates by. Defaults to [" "].
    :return: The merged template.
    """
    # parts1: new template, parts2: old template in template database
    parts1 = split_template(template1, split)
    parts2 = split_template(template2, split)

    if parts1 == parts2:
        return template1, True
    if len(parts1) == len(parts2):
        common, edit = 0, 0
        new_parts = []
        for part1, part2 in zip(parts1, parts2):
            if part1 == part2:
                common += 1
                new_parts.append(part1)
            else:
                new_part = greedy_merge_two_vars_both_side(part1, part2)
                if new_part:
                    new_parts.append(new_part)
                    edit += 1
                else:
                    new_parts.append(part1)
        if edit == 0:
            return template1, False
        new_template = ' '.join(new_parts)
        new_template = post_process_template(new_template, [])[0]
        if not validate_template(new_template):
            return template1, False
        if not verify_template_for_log_with_first_token(
                template1,
                new_template) or not verify_template_for_log_with_first_token(
                    template2, new_template):
            return template1, False
        return new_template, True
    return template1, False


def parenthesis_match(str1, str2, placeholder="\0"):
    if str1 == "(<*>)" and "(" in str2 and ")" in str2:
        return True
    return False


def colon_token(str1, placeholder="\0"):
    if str1[-1] == ":":
        return True
    return False


def func_token(str1, placeholder="\0"):
    if str1[-2:] == "()":
        return True
    return False


def judge_var_token(token1, token2, placeholder="\0"):
    """
    return True is both token1 and token2 are variables, else False if one of them is not like a variable should be merged

    """
    token1 = token1.replace("<*>", placeholder)
    token2 = token2.replace("<*>", placeholder)
    token1set, token2set = set(token1), set(token2)
    dif_token1, dif_token2 = token1set.difference(
        alphabet_set), token2set.difference(alphabet_set)

    if token1set.issubset(bigletter_set) and token2set.issubset(bigletter_set):
        return True
    if not token1set.issubset(alphabet_set) and not token2set.issubset(
            alphabet_set):
        if len(dif_token1) == 1 and len(dif_token2) == 1:
            if "." in dif_token1 and "." in dif_token2:
                return False
            if "_" in dif_token1 and "_" in dif_token2:
                return False
            if colon_token(token1) and colon_token(
                    token2) and ":" in dif_token1 and ":" in dif_token2:
                return False
        if len(dif_token1) == 2 and len(dif_token2) == 2:
            if func_token(token1) and func_token(token2):
                return False
        if "....." in token1 or "....." in token2:
            return False
        return True
    if parenthesis_match(token1, token2set, placeholder) or parenthesis_match(
            token2, token1set, placeholder):
        return True

    return False


def greedy_merge_two_vars_both_side(str1, str2):
    placeholder = "<*>"
    if str1 == "<*>" or str2 == "<*>":
        return "<*>"
    else:
        return False
    if not judge_var_token(str1, str2):
        return False

    i = 0
    while i < len(str1) and i < len(str2) and str1[i] == str2[i]:
        i += 1

    j = 0
    while j < len(str1) - i and j < len(str2) - i and str1[-(j + 1)] == str2[-(
            j + 1)]:
        j += 1

    result = []
    if i > 0:
        result.append(str1[:i])
    result.append(placeholder)
    if j > 0:
        result.append(str1[-j:])
    merged_string = ''.join(result)
    while placeholder + placeholder in merged_string:
        merged_string = merged_string.replace(placeholder + placeholder,
                                              placeholder)

    return merged_string


def merge_sorted_lists(list1, list2):
    """
    Merge two sorted lists.

    :param list1: The first sorted list.
    :param list2: The second sorted list.
    :return: The merged sorted list.
    """
    merged_list = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            if not merged_list or merged_list[-1] != list1[i]:
                merged_list.append(list1[i])
            i += 1
        elif list1[i] > list2[j]:
            if not merged_list or merged_list[-1] != list2[j]:
                merged_list.append(list2[j])
            j += 1
        else:
            if not merged_list or merged_list[-1] != list1[i]:
                merged_list.append(list1[i])
            i += 1
            j += 1

    while i < len(list1):
        if not merged_list or merged_list[-1] != list1[i]:
            merged_list.append(list1[i])
        i += 1

    while j < len(list2):
        if not merged_list or merged_list[-1] != list2[j]:
            merged_list.append(list2[j])
        j += 1

    return merged_list


class TemplateDatabase:
    """
    A class for managing a database of log templates.

    Attributes:
        None (as the __init__ method is currently empty).
    """

    def __init__(self):
        self.template_items = {}
        self.template_list = []

    def add_template(self, event_template, indexes={}, relevant_templates=[]):
        """
        Add a new template to the database.

        :param event_template: The log template to be added.
        :param indexes: A dictionary of indexes related to the template. Defaults to {}.
        :param relevant_templates: A list of relevant templates. Defaults to [].
        """
        template_tokens = split_template_naive(event_template)
        if not template_tokens or event_template == "<*>":
            return False, event_template, None, None
        if len(self.template_items) == 0 or len(template_tokens) == 1:
            self._insert_template(event_template, indexes)
            return False, event_template, None, None

        x_t = [split_template_naive(t) for t in self.template_list]
        coarse_similarities = [
            jaccard_similarity(template_tokens, t) for t in x_t
        ]

        # only compare with the most similar template
        max_sim_idx = np.argmax(coarse_similarities)
        xyz = self.template_list[max_sim_idx]
        if self._judge_template_merge_combine(event_template, xyz):
            print(f"[TemplateDB] Try Merge: `{event_template}` | `{xyz}`")
            new_template, flag_merge_success = merge_template_by_star(
                event_template, xyz)
            if flag_merge_success:
                insert_indexes = self._update_template(new_template, indexes,
                                                       max_sim_idx)
                self.template_items[new_template]['ori_templates'].append(
                    event_template)
                print(f"[TemplateDB] Merged: -> `{new_template}`")
                return True, new_template, insert_indexes, xyz
            else:
                self._insert_template(event_template, indexes)
                print(
                    f"[TemplateDB] Reject Merge, Remain Template: `{event_template}`"
                )
                return False, event_template, None, xyz
        else:
            self._insert_template(event_template, indexes)
            return False, event_template, None, xyz

    def _judge_template_merge_combine(self, template1, template2, split=[" "]):
        """
        Judge if two templates can be merged using a combined method.

        :param template1: The first template.
        :param template2: The second template.
        :param split: A list of characters to split the templates by. Defaults to [" "].
        :return: True if the templates can be merged, False otherwise.
        """
        parts1 = split_template(template1, split)
        parts2 = split_template(template2, split)
        if len(parts1) != len(parts2):
            return False
        edit_num = sum([p1 != p2 for p1, p2 in zip(parts1, parts2)])
        if edit_num <= 1:
            return True
        elif edit_num == 2 and len(parts1) > 10:
            return True
        return False

    def _insert_template(self, event_template, indexes):
        template_tokens = split_template_naive(event_template)
        self.template_items[event_template] = {
            'len': len(template_tokens),
            'indexes': indexes,
            'ori_templates': [event_template]
        }
        self.template_list.append(event_template)

    def _update_template(self, new_template, new_indexes, idx):
        old_template = self.template_list[idx]
        template_tokens = split_template_naive(new_template)

        insert_indexes = self.template_items[old_template].get('indexes',
                                                               {}).copy()
        for k, v in new_indexes.items():
            if k in insert_indexes:
                insert_indexes[k] = merge_sorted_lists(v, insert_indexes[k])
            else:
                insert_indexes[k] = v
        self.template_items[new_template] = {
            'len': len(template_tokens),
            'indexes': insert_indexes,
            'ori_templates': self.template_items[old_template]['ori_templates']
        }
        if new_template != old_template:
            self.template_items.pop(old_template)
            self.template_list.pop(idx)
            self.template_list.append(new_template)
        return insert_indexes

    def update_indexes(self, template, new_indexes):
        """
        Update the indexes of an existing template in the database.

        :param template: The log template whose indexes need to be updated.
        :param new_indexes: A dictionary of new indexes.
        """
        # old_template = self.template_list[idx]
        if template not in self.template_items:
            template_tokens = split_template_naive(template)
            self.template_items[template] = {
                'len': len(template_tokens),
                'indexes': new_indexes,
                'ori_templates': [template]
            }
            self.template_list.append(template)
            return new_indexes
        else:
            indexes2 = self.template_items[template].get('indexes', {}).copy()
            for k, v in new_indexes.items():
                if k in indexes2:
                    indexes2[k] = merge_sorted_lists(v, indexes2[k])
                else:
                    indexes2[k] = v
            print(
                f"[TemplateDB] Update Indexes: {sum(len(v) for v in self.template_items[template].get('indexes', {}).values())} -> {sum(len(v) for v in indexes2.values())} for `{template}`"
            )
            self.template_items[template]['indexes'] = indexes2
            self.template_items[template]['ori_templates'].append(template)
            return indexes2

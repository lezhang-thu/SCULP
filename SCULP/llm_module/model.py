import re
import time
from openai import OpenAI
from SCULP.llm_module.response_extractor.extract_batch import BatchExtract
from SCULP.llm_module.post_process import post_process_template
from SCULP.llm_module.template_aggregator import aggregate_by_majority
from SCULP.llm_module.variable_examples import VARIABLE_EXAMPLES_SETTING, json2prompt


class InferLLMGrouping:

    def __init__(self,
                 model,
                 api_key,
                 base_url,
                 prefix=None,
                 dataset="Apache",
                 prompt="VarExam"):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prefix = prefix
        self.dataset = dataset
        self.prompt = prompt

        self.module_params = {}
        self.messages = []
        self.usages = []

        self.system_prompt = (
            "You are the Cloud Reliability team's Log Parsing Assistant.")
        self.prompt_base_requirements = (
            "# Basic Requirements:\n"
            "- You will receive multiple log messages, each delimited by backticks.\n"
            "- Identify and replace all dynamic variables in each log with a standardized {placeholder} token, then output the static log templates.\n"
            "- Determine variable semantics and compare differences across logs to accurately identify dynamic variables.\n"
            "- Convert every occurrence of the `<*>` variable in the log into the standardized token `{variable}`. Additionally, if `<*>` appears within a literal string that matches a variable category, then, label the entire literal string as a variable.\n"
            "- Do not convert non-variables, particularly when only a single log is being parsed.\n"
            "- Paths or directories should **ALWAYS** be labeled as variables, **regardless of whether they share prefixes**. Paths or directories are processed as complete tokens, rather than substrings.\n"
            "- The above rule on labeling paths or directories are MANDOTORY. PLEASE TAKE IT SERIOUSLY!!!\n"
        )
        if "NoAdvice" not in self.prompt:
            self.prompt_variable_advice = (
                "# Advices on variables:\n"
                "- Common variables include numbers, version identifiers, IP addresses, URLs, file paths (including file names and directories), booleans, hexadecimal values, configuration keys, configuration property names, job IDs, and usernames.\n"
                "- Aim to label the entire token as a single variable. For example, replace `job-123456` with `{job_id}`, rather than `job-{job_id}`.\n"
                "- Full directories including the filename, and complex URLs (with server address or domain) must be recognized and treated as a single variable.\n"
                "- For dictionary structures, including recursively nested dictionaries, treat all values as variables, even if they are identical across logs. This rule overrides all other rules.\n"
                "- All types listed are variables, even if identical across multiple logs. **Strictly apply the corresponding `{variable_type}` replacement whenever a substring matches any listed type.**\n"
                "# Advices on non-variables:\n"
                "- Java exceptions\n"
                "- display command lines\n")
        else:
            self.prompt_variable_advice = ""

        if "NoPE" not in self.prompt:
            self.prompt_variable_example_prompt = self.construct_variable_example(
            )
            print(self.prompt_variable_example_prompt)
        else:
            self.prompt_variable_example_prompt = ""

        if "NoOutputConstraint" not in self.prompt:
            self.prompt_output_constraint = (
                "# Output Constraints:\n"
                "- Each generated template is delimited by backticks.\n"
                "- **Example Format:**\n"
                "  LogTemplate[1]: `template`\n"
                "  LogTemplate[2]: `template`\n")
        else:
            self.prompt_output_constraint = ""

        self.instruction = ""
        self.instruction += self.prompt_base_requirements
        self.instruction += self.prompt_variable_advice
        self.instruction += self.prompt_variable_example_prompt
        self.instruction += (
            "# NON-Variable Examples (these types of strings should NOT be replaced by {variaible_type}): \n"
            "- `java.io.FileNotFoundException`\n"
            "- `[auth]`\n")
        self.instruction += self.prompt_output_constraint

        print("======================== Prompt ========================")
        print(self.prompt)
        print(self.instruction)
        print("======================== Prompt ========================")

    def construct_variable_example(self):
        pe_dict = VARIABLE_EXAMPLES_SETTING['sculp']['variable_examples']
        prompt = json2prompt(pe_dict)

        return prompt

    def get_prompt_direct(self, logs, exemplars=None, proposal=None):
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": self.instruction
            },
            {
                "role": "assistant",
                "content": "OK, I'm ready to help."
            },
        ]

        query = ""
        if len(exemplars) > 0:
            query = "Example:\n"
            for e in exemplars:
                query += "Log: `{}`\nLogTemplate: `{}`\n".format(e[0], e[1])
            query += "Want u to help solve:\n"

        query_template = '\n'.join(
            [f"Log[{i+1}]: " + "`{}`" for i in range(len(logs))])
        query += query_template.format(*logs)
        if proposal:
            brain_proposal = (
                "\nLastly, the following sequence of words is likely derived from the template.\n"
                f"{proposal}")
            query += brain_proposal
        messages.append({"role": "user", "content": query})
        return messages, query

    def parsing_log_templates(self,
                              logs,
                              exemplars,
                              gts=[],
                              reparse=None,
                              proposal=None):
        # query llm for response
        messages, query = self.get_prompt_direct(logs,
                                                 exemplars=exemplars,
                                                 proposal=proposal)

        time1 = time.time()
        response = self.get_response_fallback(messages)
        query_time = time.time() - time1

        # print query
        print("\t============  Query  ====================")
        print("\n".join(["\t" + i for i in query.split('\n')]))
        # print response
        print("\t============ Response ====================")
        print(response)
        if len(gts) > 0:
            print("\t============ Target ====================")
            answer_template = '\n'.join(
                [f"\tGT Template[{i+1}]: " + "`{}`" for i in range(len(gts))])
            print(answer_template.format(*gts))

        # post process response
        try:
            gpt_templates = self.extract_and_post_process(logs, response)
            templates = [temp['post_process'] for temp in gpt_templates]
        except:
            templates = [post_process_template(log, [])[0] for log in logs]
        # aggregate templates
        best_template = aggregate_by_majority(logs, templates)

        processed2gpt = dict()
        for idx, t in enumerate(gpt_templates):
            if t['post_process'] not in processed2gpt:
                processed2gpt[t['post_process']] = (t['template'], logs[idx])

        return best_template, query_time, gpt_templates[0][
            'template'], templates, processed2gpt

    def match_log_pattern(self, template: str, log: str) -> bool:
        """
        Return True if the log matches the template pattern where
        '<*>' acts as a wildcard for any sequence of characters.
        If False, print the first place where template and log differ.
        """
        # Escape regex special characters
        regex = re.escape(template)
        # Replace '<\*>' (only * is escaped by re.escape) with wildcard
        regex = regex.replace(r'<\*>', '.*?')
        # Add anchors
        regex = '^' + regex + '$'
        match = re.match(regex, log)
        message = ""
        if match is not None:
            return True, message, regex

        # Find where it fails
        # Split template into parts: literals and wildcards
        parts = re.split(r'<\*>', template)

        pos = 0  # Current position in log
        for i, part in enumerate(parts):
            next_pos = log.find(part, pos)
            if i == 0 and next_pos != 0:
                message = (f"Mismatch: Expected log to start with '{part}'\n"
                           f"Log starts with: '{log[:min(50, len(log))]}...'")
                return False, message, regex
            elif next_pos == -1:
                message = (
                    f"Mismatch: Cannot find expected text '{part}' in log after position {pos}\n"
                    f"Log from position {pos}: '{log[pos:min(pos+50, len(log))]}...'"
                )
                return False, message, regex
            pos = next_pos + len(part)
        # Check if there's extra content at the end
        if pos < len(log):
            message = (
                f"Mismatch: Extra content at end of log\n"
                f"Expected end at position {pos}, but log continues: '{log[pos:]}'"
            )
            return False, message, regex
        return False, message, regex

    def improve_template(self, logs, template, raw_template):
        system_prompt = (
            "You are an assistant designed to refine a given template based on a set of logs. "
            "Your goal is to optimize the template so that it matches as many logs as possible."
        )
        code = (
            "    def match_log_pattern(self, template: str, log: str) -> bool:\n"
            "        \"\"\"\n"
            "        Return True if the log matches the template pattern where\n"
            "        '<*>' acts as a wildcard for any sequence of characters.\n"
            "        If False, print the first place where template and log differ.\n"
            "        \"\"\"\n"
            "        # Escape regex special characters\n"
            "        regex = re.escape(template)\n"
            "        # Replace '<\\*>' (only * is escaped by re.escape) with wildcard\n"
            "        regex = regex.replace(r'<\\*>', '.*?')\n"
            "        # Add anchors\n"
            "        regex = '^' + regex + '$'\n"
            "        match = re.match(regex, log)\n"
            "        message = \"\"\n"
            "        if match is not None:\n"
            "            return True, message, regex\n"
            "            \n"
            "        # Find where it fails\n"
            "        # Split template into parts: literals and wildcards\n"
            "        parts = re.split(r'<\\*>', template)\n"
            "        \n"
            "        pos = 0  # Current position in log\n"
            "        for i, part in enumerate(parts):\n"
            "            next_pos = log.find(part, pos)\n"
            "            if i == 0 and next_pos != 0:\n"
            "                message = (f\"Mismatch: Expected log to start with '{part}'\\n\"\n"
            "                           f\"Log starts with: '{log[:min(50, len(log))]}...'\")\n"
            "                return False, message, regex\n"
            "            elif next_pos == -1:\n"
            "                message = (\n"
            "                    f\"Mismatch: Cannot find expected text '{part}' in log after position {pos}\\n\"\n"
            "                    f\"Log from position {pos}: '{log[pos:min(pos+50, len(log))]}...'\"\n"
            "                )\n"
            "                return False, message, regex\n"
            "            pos = next_pos + len(part)\n"
            "        # Check if there's extra content at the end\n"
            "        if pos < len(log):\n"
            "            message = (\n"
            "                f\"Mismatch: Extra content at end of log\\n\"\n"
            "                f\"Expected end at position {pos}, but log continues: '{log[pos:]}'\"\n"
            "            )\n"
            "            return False, message, regex\n"
            "        return False, message, regex\n")
        t_x = self.match_log_pattern(template, logs[0])
        error_message = t_x[1]
        regex = t_x[2]
        instruction = (
            "Symbols <*> in the given template serve as wildcards representing a contiguous sequence of characters.\n"
            "You should only use <*> for matching any substring of the log text.\n"
            "Other non-<*>characters between the template and the log should exactly correspond to each other.\n"
            "At present, the given template fails to match any of the provided logs.\n"
            f"The code that performs the matching check is::\n{code}\n"
            f"The error message for the template matching the given Log[1] is:\n{error_message}\n"
            "Please present your updated template in the following format:\n"
            "ImprovedTemplate: `the updated template`\n")
        print('error_message:\n{}'.format(error_message))
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": instruction,
            },
            {
                "role": "assistant",
                "content": "OK, I'm ready to help.",
            },
        ]

        query_template = '\n'.join(
            [f"Log[{i+1}]: " + "`{}`" for i in range(len(logs))])
        query = query_template.format(*logs)
        query += '\nTemplate: `{}`'.format(template)
        if template == '':
            t_help, _ = post_process_template(raw_template, [])
            if t_help.count("<*>") > 50:
                raw_template = t_help
                msg = (
                    "Previously attempted template:\n"
                    "  `{}`\n\n"
                    "Notes:\n"
                    "- Forget about the empty template and the error message.\n"
                    "- Refine this template: it currently uses too many `<*>` placeholders. Where possible, merge them."
                ).format(raw_template)
            else:
                msg = 'Previously tried template: `{}`\n(forget about the emtpy template and the error message; try to improve over this (too general) template; we need at least one semantically meaningful natural-language word)'.format(
                    raw_template)
            print(msg)
            query += '\n{}'.format(msg)
        messages.append({"role": "user", "content": query})
        response = self.get_response_fallback(messages)
        print('#' * 30)
        print('Improving...')
        print(response)
        t = re.search(r"ImprovedTemplate:\s*`([^`]*)`", response)
        if t is None:
            return ''
        else:
            return t.group(1)

    def get_response_fallback(self, messages, temperature=0.0):
        retry_times = 0
        while retry_times < 3:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    extra_body={"thinking": {
                        "type": "disabled"
                    }})
                return response.choices[0].message.content

            except Exception as e:
                print("Exception :", e)
                if "list index out of range" in str(e):
                    break
                retry_times += 1
        return ""

    def get_compromise_response(self, logs):
        return post_process_template(logs[0], [])[0]

    def extract_and_post_process(self, logs, response):
        gpt_templates = BatchExtract.extract(response, len(logs))

        # replace null template with previous template
        gpt_templates = self.make_up_template(logs, gpt_templates)

        print("\t============ PostProcess ====================")
        for temp in gpt_templates:
            new_temp, _ = post_process_template(temp['template'], [])
            temp['post_process'] = new_temp
        return gpt_templates

    @staticmethod
    def make_up_template(logs, templates):
        """
            replace missing template with previous template
        :param logs: a list of strings
        :param templates: a list of dictionaries, each dictionary contains 'idx' and 'template'
        :return:
        """
        templates = sorted(templates, key=lambda x: x['idx'])
        # remove null template
        templates = [d for d in templates if d.get('idx') != -1]
        if len(templates) == 0:
            return [{
                'idx': i + 1,
                'template': log
            } for i, log in enumerate(logs)]

        new_templates = []
        existing_idx = [d['idx'] for d in templates]
        template_idx = -1
        for idx, _log in enumerate(logs):
            if idx + 1 not in existing_idx:
                new_templates.append({
                    'idx': idx + 1,
                    'template': templates[0]['template']
                })
            else:
                template_idx += 1
                new_templates.append({
                    'idx': idx + 1,
                    'template': templates[template_idx]['template']
                })
        new_templates = sorted(new_templates, key=lambda x: x['idx'])

        return new_templates

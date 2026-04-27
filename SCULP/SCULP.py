import os
import re
import time
import pandas as pd

from SCULP.llm_module.model import InferLLMGrouping
from SCULP.llm_module.post_process import post_process_template
from SCULP.log_partition.clustering import TopKTokenClustering
from SCULP.utils import validate_template
from SCULP.utils import preprocess_log_for_query
from SCULP.purified_template_database import TemplateDatabase


class BaseParser:

    def __init__(self,
                 add_regex,
                 regex,
                 dir_in='./',
                 dir_out='./result/',
                 rex=[],
                 data_type='full',
                 shot=0,
                 cluster_params=None,
                 llm_params=None):
        self.dir_in = dir_in
        self.dir_out = dir_out
        self.df_logs = None
        self.clusters = None
        self.add_regex = add_regex
        self.regex = regex
        self.data_type = data_type
        self.shot = shot
        self.query_count = 0
        self.max_retry_assigned = 2
        self.max_retry = 2
        self.wait_query_time = 0
        self.write_intermediate = True
        self.json_inter_result = []
        self.llm_params = llm_params
        self.cluster_params = cluster_params
        self.flag_update_template = True
        self.template_database = {}

    def save_results(self, log_name):
        to_path_logs = os.path.join(
            self.dir_out, f"{log_name}_{self.data_type}.log_structured.csv")
        df_to_save = self.clusters.prepare_save_df(self.gt_parsed)
        #df_to_save.to_csv(to_path_logs, index=False)
        print(f"Saved {log_name}_log_structured.csv to {to_path_logs}")

        to_path_templates = os.path.join(
            self.dir_out, f"{log_name}_{self.data_type}.log_templates.csv")
        df_templates = df_to_save.loc[:, ['EventId', 'EventTemplate'
                                         ]].drop_duplicates()
        df_templates['EventId_numeric'] = df_templates['EventId'].str.extract(
            '(\d+)').astype(int)
        df_selected_sorted = df_templates.sort_values(by='EventId_numeric')
        df_selected_sorted = df_selected_sorted.drop('EventId_numeric', axis=1)

        #df_selected_sorted.to_csv(to_path_templates, index=False)
        print(f"Saved {log_name}_log_templates.csv to {to_path_templates}")
        self.gpt_parsed = df_to_save

    def validate_and_update_with_cluster_map_template_database(
            self, logs_to_query, template, cluster_id):
        time1 = time.time()
        update_success, update_num = False, 0
        new_template = None
        if validate_template(template):
            update_success, update_num, updated_indexes = self.clusters.update_logs_with_map(
                template, cluster_id)
            if update_success:
                parent_cluster_id = self.clusters.update_map_child2parent[
                    cluster_id]
                need_update, new_template, insert_indexes, similar_template = self.template_database[
                    parent_cluster_id].add_template(template, updated_indexes)
                if need_update and validate_template(new_template):
                    update_num = self.clusters.update_logs_by_indexes(
                        new_template, cluster_id, insert_indexes)

                    if similar_template in self.post2template:
                        del self.examples[self.post2template[similar_template]]
                        del self.post2template[similar_template]

                    if new_template != template:
                        _, update_num, updated_indexes = self.clusters.update_logs_with_map(
                            new_template, cluster_id)
                        self.template_database[
                            parent_cluster_id].update_indexes(
                                new_template, updated_indexes)
                        print(
                            f"[TemplateBaseUpdate] Match unparsed logs {update_num} with new template `{new_template}`"
                        )
                print(
                    f"Update Success: Time for one update logs: {time.time() - time1}, template `{template}`"
                )
            else:
                print(
                    f"Update failed: Template can not match logs `{template}`. Retry query"
                )
        else:
            print(
                f"Update failed: Validate template `{template}` failed. Retry query"
            )

        return update_success, update_num, new_template


class SCULPParser(BaseParser):

    def __init__(self,
                 add_regex,
                 regex,
                 dir_in='./',
                 dir_out='./result/',
                 rex=[],
                 data_type='full',
                 shot=0,
                 cluster_params=None,
                 llm_params=None):
        super().__init__(add_regex, regex, dir_in, dir_out, rex, data_type,
                         shot, cluster_params, llm_params)
        self.llm = InferLLMGrouping(**self.llm_params)
        self.clusters = TopKTokenClustering(
            sample_method=self.cluster_params["sample_method"],
            sample_size=self.cluster_params["sample_size"],
            min_cluster_size=self.cluster_params["min_cluster_size"],
            cluster_topk=self.cluster_params["cluster_topk"],
            sample_min_similarity=self.cluster_params["sample_min_similarity"],
            lcu_lamb=self.cluster_params["lcu_lamb"],
            lcu_sample_size=self.cluster_params["lcu_sample_size"],
            sample_size_auto=self.cluster_params["sample_size_auto"],
            add_regex=self.cluster_params["add_regex"],
            regex=self.cluster_params["regex"],
            pad_query=self.cluster_params["pad_query"],
        )

        self.examples = {}
        self.post2template = {}

    def parse(self, logName):
        log_path = os.path.join(
            self.dir_in, f"{logName}_{self.data_type}.log_structured.csv")
        print('Parsing file: ' + log_path)
        self.gt_parsed = pd.read_csv(log_path)
        self.clusters.load_data(self.gt_parsed, log_path)
        logs_grouped = self.clusters.clustering()
        self.initialize_template_database()

        n_iter = 0
        while self.clusters.num_processed_logs < self.clusters.num_total_logs:
            # Sample logs to query
            print("Iteration {}".format(n_iter))
            prev_templates = list()
            update_success, logs_to_query, logs_to_query_regex, template, cluster_id, wrong_template, all_templates, processed2gpt = self.parse_one_iter(
                reparse=prev_templates)
            prev_templates.append(wrong_template)

            if not update_success:
                print("Update failed. Try to get a compromise response")
                template = self.llm.get_compromise_response(logs_to_query_regex)
                update_success, update_num, _ = self.validate_and_update_with_cluster_map_template_database(
                    logs_to_query_regex, template, cluster_id)
            if update_success and len(all_templates) > 0:
                print(
                    "A good starting point. Try to use the remaining templates..."
                )
                for x_template in all_templates:
                    print(x_template)
                    flag, _, new_template = self.validate_and_update_with_cluster_map_template_database(
                        logs_to_query_regex, x_template, cluster_id)
                    if flag:
                        self._choose(new_template, x_template,
                                     processed2gpt[x_template][0],
                                     processed2gpt[x_template][1])
            if not update_success:
                print(f"Update failed. Get a compromise response also failed.")
                update_success, update_num, _ = self.validate_and_update_with_cluster_map_template_database(
                    logs_to_query_regex, logs_to_query_regex[0], cluster_id)
            print(
                "========================================================================================\n\n"
            )
            n_iter += 1
            if len(logs_to_query) > 0:
                save_item = {
                    "iter": n_iter,
                    "logs_to_query": logs_to_query,
                    "logs_to_query_regex": logs_to_query_regex,
                    "llm_template": template,
                    "cluster_id": int(cluster_id),
                    "update_success": update_success,
                }
                self.json_inter_result.append(save_item)
        self.save_results(logName)

    def _to_map(self, template, raw_format, log):
        self.examples[raw_format] = log
        self.post2template[template] = raw_format

    def _choose(self, new_template, template, raw_format, log):
        if new_template and new_template != template:
            self._to_map(new_template,
                         new_template.replace("<*>", "{variable}"), log)
        else:
            self._to_map(template, raw_format, log)

    def parse_one_iter(self, reparse):
        cluster_id, logs_to_query, proposal_template = self.clusters.sample_for_llm(
        )
        print("cluster_id: {}\nlogs_to_query: {}".format(
            cluster_id, logs_to_query))
        print('self.add_regex: {}'.format(self.add_regex))
        if self.add_regex == "add":
            logs_to_query_regex = [
                preprocess_log_for_query(log, self.regex)
                for log in logs_to_query
            ]
        else:
            logs_to_query_regex = logs_to_query
        logs_to_query_regex = [
            re.sub(r'`[^`]*`', "{variable}", log) for log in logs_to_query_regex
        ]
        logs_to_query_regex = [
            re.sub(r"`[^']*'", "{variable}", log) for log in logs_to_query_regex
        ]

        # Query LLM
        examplars = []
        template, query_time, wrong_template, all_templates, processed2gpt = self.llm.parsing_log_templates(
            logs_to_query_regex,
            examplars,
            reparse=reparse,
            proposal=proposal_template)
        self.wait_query_time += query_time
        self.query_count += 1
        print("\t============ Aggregate ====================")
        print("\tAggregated Template: ", template)
        update_success, update_num, new_template = self.validate_and_update_with_cluster_map_template_database(
            logs_to_query_regex, template, cluster_id)
        if update_success:
            self._choose(new_template, template, processed2gpt[template][0],
                         processed2gpt[template][1])
        counter = 0

        required_log = processed2gpt[template][
            1] if template in processed2gpt else next(
                iter(processed2gpt.values()))[1]
        gpt_raw = re.sub(
            r'\{[A-Za-z0-9_-]+\}', '<*>',
            processed2gpt[template][0] if template in processed2gpt else next(
                iter(processed2gpt.values()))[0])
        while not update_success and counter < 3:
            llm_template = self.llm.improve_template(
                logs_to_query_regex,
                template,
                gpt_raw,
            )
            llm_template, correct = post_process_template(llm_template, [])
            if correct:
                print("llm_template:\n{}".format(llm_template))
                template = llm_template
                update_success, update_num, new_template = self.validate_and_update_with_cluster_map_template_database(
                    logs_to_query_regex, template, cluster_id)
                if update_success:
                    self._choose(new_template, template,
                                 template.replace("<*>", "{variable}"),
                                 required_log)
            counter += 1

        t = set() if counter > 0 else set(all_templates) - set([template])
        return update_success, logs_to_query, logs_to_query_regex, template, cluster_id, wrong_template, t, processed2gpt

    def initialize_template_database(self):
        for k, _df in self.clusters.clusters.items():
            cid = _df["cid1"].iloc[0]
            if cid not in self.template_database:
                self.template_database[cid] = TemplateDatabase()

import re
import math
import random
import pandas as pd
from collections import Counter

pd.set_option('mode.chained_assignment', None)
from SCULP.utils import verify_template_for_log_with_first_token
from SCULP.utils import preprocess_log_for_query


class BaseClustering:

    def __init__(self,
                 sample_method="lcu_sampling",
                 sample_size=3,
                 min_cluster_size=100,
                 sample_min_similarity=0.5,
                 lcu_lamb=0.5,
                 lcu_sample_size=3,
                 sample_size_auto="fixed",
                 add_regex="add",
                 regex=[],
                 add_skip_sim=False,
                 pad_query=True):
        self.df_logs = None
        self.log_path = None
        self.num_total_logs = 0
        self.num_processed_logs = 0
        self.add_regex = add_regex
        self.regex = regex
        self.sample_method = sample_method
        self.min_cluster_size = min_cluster_size
        self.sample_min_similarity = sample_min_similarity
        self.lcu_lamb = lcu_lamb
        self.lcu_sample_size = lcu_sample_size
        self.sample_size = sample_size
        self.sample_size_assigned = sample_size
        self.max_sample_size, self.min_sample_size = 5, 1
        self.max_log_length, self.min_log_length = -1, -1
        if sample_size_auto == "auto":
            self.sample_size_auto = True
        else:
            self.sample_size_auto = False
        self.pad_query = pad_query
        self.add_skip_sim = add_skip_sim
        self.vectors = []
        self.log_lengths = []
        self.clusters = {}
        self.current_logs_bucket_id = -1
        self.current_logs_bucket = pd.DataFrame(
            columns=["LineId", "Content", "EventId", "Template"])
        self.update_map_parent2child = {}
        self.update_map_child2parent = {}

    def load_data(self, df_logs, log_path):
        print("Clustering load data")
        self.log_path = log_path
        self.df_logs = df_logs
        # debug
        print('self.add_regex: {}'.format(self.add_regex))
        print('self.regex: {}'.format(self.regex))
        if self.add_regex == "before":
            print("Clustering add regex before preprocess")
            self.df_logs.loc[:, "Content"] = self.df_logs.apply(
                lambda row: preprocess_log_for_query(row["Content"], self.regex
                                                    ),
                axis=1)

        self.df_logs = self.df_logs.assign(Template="")
        print('self.df_logs.iloc[23]:\n{}'.format(self.df_logs.iloc[23]))
        self.num_total_logs = len(self.df_logs)
        self.num_processed_logs = 0

    def prepare_save_df(self, original_df_logs):
        # factorize Template → 0..n-1
        codes = self.df_logs["Template"].astype("category").cat.codes
        self.df_logs["NewEventId"] = "E" + (codes + 1).astype(str)

        df = self.df_logs[["LineId", "Content", "NewEventId", "Template"]]
        df["Content"] = original_df_logs["Content"]
        df.columns = ["LineId", "Content", "EventId", "EventTemplate"]
        return df

    def update_logs_with_map(self, template, child_id):
        if template == "":
            print("Fail to update Template is empty")
            return [], 0, {}
        parent_id = self.update_map_child2parent[child_id]
        bucket_ids_to_check = self.update_map_parent2child[parent_id]
        index = []
        all_indexes = {}
        total_matched = 0
        total_num_before, total_num_after = 0, 0
        for bucket_id in bucket_ids_to_check:
            current_logs_bucket = self.clusters[bucket_id]
            num_berfore = len(current_logs_bucket)
            current_logs_bucket.loc[:, "Matched"] = current_logs_bucket.apply(
                lambda row: verify_template_for_log_with_first_token(
                    row["Content"], template),
                axis=1)
            index = current_logs_bucket[current_logs_bucket["Matched"] ==
                                        True].index
            self.num_processed_logs += len(index)
            self.df_logs.loc[index, "Template"] = template
            current_logs_bucket = current_logs_bucket.loc[
                current_logs_bucket["Matched"] == False]
            self.clusters[bucket_id] = current_logs_bucket
            num_after = len(current_logs_bucket)
            total_matched += num_berfore - num_after
            total_num_before += num_berfore
            total_num_after += num_after
            all_indexes[bucket_id] = index.tolist()
        empty_bucket_num = len(
            [i for i in self.clusters.values() if len(i) != 0])
        print(
            f"[UpdateBucket] Logs: This iter found: {total_matched}, total: {self.num_processed_logs}/{self.num_total_logs}, "
            f"remain: {self.num_total_logs-self.num_processed_logs}. ")
        print(
            f"[UpdateBucket] Buckets: Checked {len(bucket_ids_to_check)} ({bucket_ids_to_check}), Parent Bucket size: {total_num_before} -> {total_num_after}, remain buckets: {empty_bucket_num}"
        )
        if total_matched == 0:
            return False, 0, {}
        return True, total_matched, all_indexes

    def update_logs_by_indexes(self, template, child_id, all_indexes):
        if template == "":
            print(
                "[TemplateBaseUpdate] Fail to modify Template from an empty template"
            )
            return 0
        if not all_indexes:
            print(
                "[TemplateBaseUpdate] No existing indexes to check and update")
            return 0
        parent_id = self.update_map_child2parent[child_id]
        bucket_ids_to_check = self.update_map_parent2child[parent_id]
        total, total_updated = 0, 0
        for bucket_id in bucket_ids_to_check:
            index = pd.Index(all_indexes[bucket_id])
            rows_to_process = self.df_logs.loc[index]
            verify_results = rows_to_process.apply(
                lambda row: verify_template_for_log_with_first_token(
                    row["Content"], template),
                axis=1)
            index_to_update = verify_results[verify_results == True].index
            self.df_logs.loc[index_to_update, "Template"] = template
            total_updated += len(index_to_update)
            total += len(index)
        print(
            f"[TemplateBaseUpdate] Update previous logs with merged template, succeed/all: {total_updated}/{total}, in child Bucket {bucket_ids_to_check}"
        )
        return total_updated

    def sample_by_lcu_sampling(self, dedup=True):
        if len(self.clusters) == 0:
            self.clustering()
        # Strategy: always sample the largest cluster
        max_cluster_id = max(self.clusters, key=lambda k: len(self.clusters[k]))

        self.current_logs_bucket_id = max_cluster_id
        self.current_logs_bucket = self.clusters[self.current_logs_bucket_id]
        proposal_template = None
        print(
            f"Sample {self.sample_size} from current logs bucket: ID: {self.current_logs_bucket_id}, Len: {self.current_logs_bucket['length'].iloc[0]}, Bucket Size: {len(self.current_logs_bucket)}, Total Buckets: {len(self.clusters)}",
        )

        if len(self.current_logs_bucket) == 0:
            print('A')
            assert False, "The branch should NOT be entered!"
            self.clusters.pop(self.current_logs_bucket_id)
            cluster_id, sampled = self.sample_by_lcu_sampling(dedup=dedup)
            return cluster_id, sampled, proposal_template
        elif len(self.current_logs_bucket) == 1:
            print('B')
            logs = self.current_logs_bucket["Content"].tolist()
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            return cluster_id, logs, None
        else:
            print('C')
            candidate_logs = self.current_logs_bucket[
                "Content"].drop_duplicates().tolist()
            print("len(candidate_logs): {}".format(len(candidate_logs)))
            if len(candidate_logs) == 1:
                proposal_template = None
            cluster_id = self.current_logs_bucket["cid2"].iloc[0]
            return cluster_id, get_diverse_anchors(candidate_logs, 5), None


class TopKTokenClustering(BaseClustering):
    """ Very similar to Drain with a depth 5"""

    def __init__(self,
                 sample_method="lcu_sampling",
                 sample_size=3,
                 cluster_topk=3,
                 min_cluster_size=100,
                 sample_min_similarity=0.5,
                 lcu_lamb=0.5,
                 lcu_sample_size=3,
                 sample_size_auto="fixed",
                 add_regex="add",
                 regex=[],
                 add_skip_sim=False,
                 pad_query=True):
        super(TopKTokenClustering,
              self).__init__(sample_method,
                             sample_size,
                             min_cluster_size=min_cluster_size,
                             sample_min_similarity=sample_min_similarity,
                             lcu_lamb=lcu_lamb,
                             lcu_sample_size=lcu_sample_size,
                             sample_size_auto=sample_size_auto,
                             add_regex=add_regex,
                             regex=regex,
                             add_skip_sim=add_skip_sim,
                             pad_query=pad_query)
        self.cluster_topk = cluster_topk
        self.token_frequency = Counter()

    def represent(self):
        self.df_logs["length"] = self.df_logs["Content"].apply(
            get_tokens_length)
        self.log_lengths = self.df_logs["length"].tolist()

    def clustering(self):
        if len(self.log_lengths) == 0:
            self.represent()
        df_logs = self.df_logs[self.df_logs["Template"] == ""]
        grouped = df_logs.groupby("length").groups
        self.max_log_length, self.min_log_length = max(grouped.keys()), min(
            grouped.keys())

        # Cluster by log length
        _bucket_to_merge = {}
        for idx, key in enumerate(sorted(grouped.keys())):
            this_bucket = self.df_logs.iloc[grouped[key]]
            _bucket_to_merge[idx] = this_bucket
        self.clusters = _bucket_to_merge
        print(f"Clustering by log length: {len(self.clusters)}")
        print(
            f"Clustering by log length: {[len(i) for i in self.clusters.values()]}"
        )

        # Cluster by top-k tokens
        flat_clusters = {}
        for idx, cluster in self.clusters.items():
            _clusters = self.brain_cluster(cluster)
            cid2 = len(flat_clusters)
            for i, df in enumerate(_clusters):
                df.loc[:, "cid1"] = [idx] * len(df)
                df.loc[:, "cid2"] = [cid2 + i] * len(df)
            for child_idx in range(len(flat_clusters),
                                   len(flat_clusters) + len(_clusters)):
                self.update_map_child2parent[child_idx] = idx
            self.update_map_parent2child[idx] = list(
                range(len(flat_clusters),
                      len(flat_clusters) + len(_clusters)))
            for _clus in _clusters:
                flat_clusters[len(flat_clusters)] = _clus
        print(
            f"Clustering (min_cluster_size={self.min_cluster_size}) by length and 1st 3 tokens: {len(flat_clusters)} clusters"
        )
        self.clusters = flat_clusters

        # Merge small clusters
        print(f"Clustering results: {[len(i) for i in self.clusters.values()]}")

        return self.clusters

    def sample_for_llm(self):
        return self.sample_by_lcu_sampling(dedup=True)

    def brain_cluster(self, df):
        # 1. Identify Unique Content
        # We operate solely on the unique patterns first to save computation
        unique_content = df["Content"].drop_duplicates()

        # 2. Split into tokens (Only for unique rows)
        unique_token_lists = unique_content.str.split()

        # Assert same token length
        # (We check lengths on unique_token_lists; if unique ones are consistent, all are)
        lengths = unique_token_lists.str.len()
        if len(lengths) > 0:
            assert lengths.nunique(
            ) == 1, "All rows must have same number of tokens"

        # 3. Build Token Matrix for UNIQUE rows
        # We use .reset_index(drop=True) on the temporary dataframe to align
        # the numpy arrays later, while keeping unique_content mapping safe.
        unique_token_df = pd.DataFrame(unique_token_lists.tolist())

        # 4. Compute column-wise token frequencies
        # (This logic is inherently based on unique patterns per your original code)
        freq_lookup = {}
        for col in unique_token_df.columns:
            freq_lookup[col] = unique_token_df[col].value_counts().to_dict()

        # 5. Map frequencies to the UNIQUE Matrix
        unique_freq_df = unique_token_df.copy()
        for col in unique_token_df.columns:
            unique_freq_df[col] = unique_token_df[col].map(freq_lookup[col])

        # 6. Compute features for UNIQUE rows
        # We extract values to numpy arrays for faster iteration than .iterrows()
        token_values = unique_token_df.values
        freq_values = unique_freq_df.values

        unique_features = []

        # Iterate through unique patterns only
        for i in range(len(token_values)):
            tokens = token_values[i]
            freqs = freq_values[i]

            # --- Original Logic Logic ---
            freq_counts = Counter(freqs)
            max_count = max(freq_counts.values())

            # Tie-breaking logic
            tied_freqs = [f for f, c in freq_counts.items() if c == max_count]
            most_common_freq = max(tied_freqs)

            if most_common_freq == 1:
                most_common_freq = max(freqs)

            # Select tokens corresponding to the winner frequency
            # zip is faster here than index lookup
            feature_tokens = tuple(
                t for t, f in zip(tokens, freqs) if f == most_common_freq)
            unique_features.append(feature_tokens)

        # 7. Map features back to the Original DataFrame
        # Create a hash map: { "Content String" : ("Feature", "Tuple") }
        content_to_feature_map = dict(zip(unique_content, unique_features))

        df = df.copy()
        df["_feature"] = df["Content"].map(content_to_feature_map)

        # 8. Split df by identical features
        grouped_dfs = [group for _, group in df.groupby("_feature")]

        return grouped_dfs


def get_tokens(log, separator=[" "]):
    for sep in separator:
        log = log.replace(sep, " ")
    return log.split()


def get_tokens_length(log, separator=[" "]):
    return len(get_tokens(log, separator))


def get_diverse_anchors(candidate_logs, n_anchors=5):
    """
    Selects n_anchors from candidate_logs that are least similar to each other
    (maximally diverse).

    Algorithm: Greedy Max-Min
    1. For every point, track its similarity to its *nearest* selected anchor.
    2. Pick the point that has the *lowest* similarity to its nearest anchor.
    """
    if len(candidate_logs) <= 1:
        return candidate_logs

    n = len(candidate_logs)

    # Convert strings to sets once to avoid repeated splitting/hashing.
    candidate_sets = [set(log.split()) for log in candidate_logs]

    # Indices of the chosen anchors
    anchor_indices = [0]
    selected_indices_set = {0}

    # Internal helper to calculate Jaccard of one anchor against all candidates
    def compute_jaccard_vector(anchor_idx):
        anchor_set = candidate_sets[anchor_idx]
        len_anchor = len(anchor_set)
        sims = []

        for other_set in candidate_sets:
            # Union = len(A) + len(B) - Intersection
            intersection_size = len(anchor_set.intersection(other_set))
            union_size = len_anchor + len(other_set) - intersection_size

            if union_size == 0:
                sims.append(0.0)
            else:
                sims.append(intersection_size / union_size)
        return sims

    # Initialize state with the first anchor (index 0)
    # This list tracks, for every candidate, the HIGHEST similarity score
    # it has with any of the currently selected anchors.
    max_sim_to_closest_anchor = compute_jaccard_vector(0)

    # Mark the first anchor as "infinite" similarity so it isn't picked again.
    # (Since we look for the MIN value later, INF effectively removes it from consideration)
    max_sim_to_closest_anchor[0] = math.inf

    def random_argmin(values):
        """Returns a random index among those with the minimum value."""
        min_val = min(values)
        # Find all indices that share this minimum value
        candidates = [i for i, v in enumerate(values) if v == min_val]
        return random.choice(candidates)

    for _ in range(1, min(n_anchors, n)):

        # 1. Selection Step:
        # Find the candidate that is *least* similar to the current group of anchors.
        # i.e., Minimize the Maximum similarity.
        next_idx = random_argmin(max_sim_to_closest_anchor)

        anchor_indices.append(next_idx)
        selected_indices_set.add(next_idx)

        # 2. Update Step:
        # Calculate similarity of all candidates to this NEW anchor
        new_anchor_sims = compute_jaccard_vector(next_idx)

        # Update the tracker
        for i in range(n):
            if i in selected_indices_set:
                max_sim_to_closest_anchor[i] = math.inf
            else:
                # We update the score if the NEW anchor is closer (more similar)
                # to candidate i than any previous anchor was.
                if new_anchor_sims[i] > max_sim_to_closest_anchor[i]:
                    max_sim_to_closest_anchor[i] = new_anchor_sims[i]

    # Map selected indices back to original strings
    return [candidate_logs[i] for i in anchor_indices]

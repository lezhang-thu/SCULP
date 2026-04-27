import os
import time
import csv
import pandas as pd
import numpy as np
from pathlib import Path

from .evaluator_ga import compute_grouping_accuracy
from .evaluator_fta import compute_template_level_accuracy
from .evaluator_pa import calculate_parsing_accuracy
from SCULP.llm_module.post_process import correct_single_template


def prepare_results(output_dir, otc, complex, frequent):
    if not os.path.exists(output_dir):
        # make output directory
        os.makedirs(output_dir)

    # make a new summary file
    result_file = 'summary_[otc={},complex={},frequent={}].csv'.format(
        str(otc), str(int(complex)), str(int(frequent)))
    if not os.path.exists(os.path.join(output_dir, result_file)):
        with open(os.path.join(output_dir, result_file), 'w') as csv_file:
            fw = csv.writer(csv_file,
                            delimiter=',',
                            quotechar='|',
                            quoting=csv.QUOTE_MINIMAL)
            # fw.writerow(['Dataset', 'GA_time', 'PA_time', 'TA_time', 'parse_time', 'identified_templates',
            #              'ground_templates', 'GA', 'PA', 'FTA', 'PTA', 'RTA', 'OG', 'UG', 'MX'])
            fw.writerow([
                'Dataset', 'identified_templates', 'ground_templates', 'GA',
                'PA', 'FGA', 'PTA', 'RTA', 'FTA'
            ])

    return result_file


def evaluator(
    dataset,
    data_type,
    input_dir,
    output_dir,
    result_file,
    otc=False,
    complex=False,
    frequent=False,
    groundtruth=None,
    parsedresult=None,
):
    print('\n=== Evaluation on %s ===' % dataset)
    if otc:
        # use a structured log file with corrected oracle templates
        #groundtruth = os.path.join(input_dir, f"{dataset}_{data_type}.log_structured_corrected.csv")
        print("corrected oracle templates")
    else:
        #groundtruth = os.path.join(input_dir, f"{dataset}_{data_type}.log_structured.csv")
        gt_template = os.path.join(input_dir,
                                   f"{dataset}_{data_type}.log_templates.csv")
    parsedresult.fillna("", inplace=True)
    gt_template = pd.read_csv(gt_template, dtype=str)

    # Step 1: Build dict from gt_template
    id_to_template = dict(
        zip(gt_template["EventId"], gt_template["EventTemplate"]))

    # Step 2: Correct templates ONCE
    corrected_dict = {
        eid: correct_single_template(tpl) for eid, tpl in id_to_template.items()
    }

    # Step 3: Apply corrected templates to groundtruth using EventId
    groundtruth["EventTemplate"] = groundtruth["EventId"].map(corrected_dict)

    filter_templates = None
    if complex != 0:
        print("Evaluate on complex mode: ", complex)
        template_file = os.path.join(
            input_dir, f"{dataset}_{data_type}.log_templates.csv")
        df = pd.read_csv(template_file)
        df = template_file
        if complex == 1:
            df = df[df['EventTemplate'].str.count('<*>') == 0]
        if complex == 2:
            df = df[(df['EventTemplate'].str.count('<*>') >= 1) &
                    (df['EventTemplate'].str.count('<*>') <= 4)]
        if complex == 3:
            df = df[df['EventTemplate'].str.count('<*>') >= 5]
        filter_templates = df['EventTemplate'].tolist()

    if frequent != 0:
        print("Evaluate on frequent mode: ", frequent)
        template_file = os.path.join(
            input_dir, f"{dataset}_{data_type}.log_templates.csv")
        df = pd.read_csv(template_file)
        df = template_file
        df_sorted = df.sort_values('Occurrences')
        if frequent > 0:
            n = int(len(df_sorted) / 100.0 * frequent)
            filter_templates = df_sorted['EventTemplate'].tolist()[:n]
        else:
            n = len(df_sorted) - int(len(df_sorted) / 100.0 * -frequent)
            filter_templates = df_sorted['EventTemplate'].tolist()[n:]

    if filter_templates != None:
        print("length of filter templates: ", len(filter_templates))

    if filter_templates != None and len(filter_templates) == 0:
        result = dataset + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + ',' + \
                 "None" + '\n'

        with open(os.path.join(os.path.dirname(output_dir), result_file),
                  'a') as summary_file:
            summary_file.write(result)
        return

    print("Start compute grouping accuracy")
    # calculate grouping accuracy
    start_time = time.time()
    GA, FGA = compute_grouping_accuracy(groundtruth, parsedresult,
                                        filter_templates)
    GA_end_time = time.time() - start_time
    print('Grouping Accuracy calculation done. [Time taken: {:.3f}]'.format(
        GA_end_time))

    # calculate parsing accuracyg
    start_time = time.time()
    PA = calculate_parsing_accuracy(groundtruth, parsedresult, filter_templates)
    PA_end_time = time.time() - start_time
    print('Parsing Accuracy calculation done. [Time taken: {:.3f}]'.format(
        PA_end_time))

    # calculate template-level accuracy
    start_time = time.time()
    tool_templates, ground_templates, FTA, PTA, RTA = compute_template_level_accuracy(
        dataset, groundtruth, parsedresult, filter_templates)
    TA_end_time = time.time() - start_time
    print(
        'Template-level accuracy calculation done. [Time taken: {:.3f}]'.format(
            TA_end_time))

    result = dataset + ',' + \
             str(tool_templates) + ',' + \
             str(ground_templates) + ',' + \
             "{:.3f}".format(GA) + ',' + \
             "{:.3f}".format(PA) + ',' + \
             "{:.3f}".format(FGA) + ',' + \
             "{:.3f}".format(PTA) + ',' + \
             "{:.3f}".format(RTA) + ',' + \
             "{:.3f}".format(FTA) + '\n'

    with open(os.path.join(os.path.dirname(output_dir), result_file),
              'a') as summary_file:
        summary_file.write(result)


def post_average(metric_file, output_path):
    df = pd.read_csv(metric_file, index_col=False)
    df = df.drop_duplicates(['Dataset'])
    mean_row = df.select_dtypes(include=[np.number]).mean().round(3)
    new_row = pd.DataFrame([['Average']], columns=['Dataset']).join(
        pd.DataFrame([mean_row.values], columns=mean_row.index))
    df = pd.concat([df, new_row], ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    df = pd.read_csv(output_path)
    transposed_df = df.transpose()
    transposed_df.to_csv(output_path)

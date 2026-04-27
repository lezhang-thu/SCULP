import os
import time
import random
import queue
import threading

from SCULP.SCULP_parallel import SCULPParserParallel
from SCULP.config import load_args
from utils.evaluator_main import evaluator, prepare_results, post_average


def worker(q, parser):
    while True:
        # Get a task from the queue
        task_id = q.get()

        try:
            # Process the task
            parser.parse(task_id)
            # Check if it needs to go back in
            if parser.non_empty(task_id):
                q.put(task_id)
        except Exception as e:
            print(f"[Worker] Exception on task {task_id}: {e}")
            raise  # optional: re-raise if you want to crash

        # Notify the queue that one unit of work is finished
        q.task_done()


random.seed(22222)
if __name__ == "__main__":
    random.seed(22222)
    print("Start to parse logs")
    args = load_args()

    input_dataset_dir = os.path.join(args.data_dir, args.test_dataset)
    output_dataset_dir = os.path.join(args.output_dir, args.test_dataset)
    print(f"Input dir: {input_dataset_dir}")
    print(f"Output dir: {output_dataset_dir}")
    if not os.path.exists(output_dataset_dir):
        os.makedirs(output_dataset_dir)

    time_start = time.time()
    parser = SCULPParserParallel(
        add_regex=args.add_regex,
        regex=args.regex,
        data_type=args.data_type,
        dir_in=input_dataset_dir,
        dir_out=output_dataset_dir,
        rex=[],
        llm_params=args.llm_params,
        cluster_params=args.cluster_params,
    )

    # Hierarchical Sharding
    time_start_after_load = parser.init_cluster(args.test_dataset)

    hyperbucket_queue = queue.Queue()
    _ = parser.clusters.clustering()
    for k in range(len(parser.clusters.update_map_parent2child)):
        hyperbucket_queue.put(k)

    num_threads = 8
    # 3. Start the worker threads
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker,
                             args=(hyperbucket_queue, parser),
                             name=f"Worker-{i}")
        t.daemon = True  # Allows the program to exit even if threads are running
        t.start()
        threads.append(t)

    # 4. Wait until queue empty AND no tasks in progress
    hyperbucket_queue.join()
    print("\nAll tasks finished.")

    finish_parsing_time = time.time()
    parser.save_results(args.test_dataset)
    print(parser.dir_out)
    output = []

    print(
        f"Total parsing time: {finish_parsing_time - time_start_after_load} seconds (no input output)"
    )
    print(
        f"Total parsing time: {finish_parsing_time - time_start} seconds (no output)"
    )
    print(
        f"Total parsing time: {time.time() - time_start} seconds (with output)")

    result_file = prepare_results(args.output_dir,
                                  otc=args.otc,
                                  complex=args.complex,
                                  frequent=args.frequent)
    evaluator(
        args.test_dataset,
        args.data_type,
        input_dataset_dir,
        output_dataset_dir,
        result_file,
        otc=args.otc,
        complex=args.complex,
        frequent=args.frequent,
        groundtruth=parser.gt_parsed,
        parsedresult=parser.gpt_parsed,
    )
    post_average(
        os.path.join(args.output_dir, result_file),
        os.path.join(
            args.output_dir,
            f"SCULP_{args.data_type}_complex={args.complex}_frequent={args.frequent}_{args.model}.csv"
        ))

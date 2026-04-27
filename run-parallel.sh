#!/bin/bash

base_output_dir="./saved_results/sculp-parallel"

if [ "$1" == "all" ]; then
	datasets=("Proxifier" "Apache" "OpenSSH" "HDFS" "OpenStack" "HPC" "Zookeeper" "HealthApp" "Hadoop" "Spark" "BGL" "Linux" "Mac" "Thunderbird")
	shift
else
	datasets=("$@")
fi

for run in 0; do
	#for run in 1 2; do
	OUTPUT_DIR="${base_output_dir}" # Reset to base dir before each run
	echo "=== Starting run ${run} ==="

	for dataset in "${datasets[@]}"; do
		echo "Running on ${dataset} (run ${run})"
		mkdir -p "${OUTPUT_DIR}/${dataset}"
		python -u parallel-entry.py --test_dataset "${dataset}" --output_dir ${OUTPUT_DIR} \
			2>&1 | tee "${OUTPUT_DIR}/${dataset}/log_test.txt"
	done

	# Rename the output directory after the run
	mv "${base_output_dir}" "${base_output_dir}-${run}"
done

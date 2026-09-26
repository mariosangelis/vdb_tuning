import json

# Fields to average
fields_to_average = [
    "total_exec_time_topk",
    "total_exec_time_tpch",
    "shared_hit_tpch",
    "shared_read_tpch",
    "shared_read_topk",
    "shared_hit_topk"
]

# Initialize sums and count
sums = {field: 0.0 for field in fields_to_average}
count = 0

# Read and process file
with open("mixed_workload_TPCH_AND_TOPK_BATCHED_BAS_NORMAL_8G_SHARED_BUFFERS_WIT_PREDICATE.jsonl", "r") as f:
    for line in f:
        try:
            entry = json.loads(line)
            for field in fields_to_average:
                sums[field] += entry.get(field, 0.0)
            count += 1
        except Exception as e:
            print("Skipping line due to error:", e)

# Compute and print averages
print(f"Processed {count} entries.\n")
for field in fields_to_average:
    avg = sums[field] / count if count else 0.0
    print(f"Average {field}: {avg:.4f}")

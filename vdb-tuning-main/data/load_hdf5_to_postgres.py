#!/usr/bin/env python3
"""
Load HDF5 vector datasets into PostgreSQL with pgvector.
Uses efficient COPY BINARY method (same as BigVectorBench).

Usage:
    python3 load_hdf5_to_postgres.py --file sift10m-6filter-6a.hdf5 --table sift_data
    python3 load_hdf5_to_postgres.py --file sift10m-6filter-6a.hdf5 --table sift_data --create-index hnsw
"""

import argparse
import h5py
import numpy as np
import time
import json
import psycopg
import pgvector.psycopg

# Database connection settings (matching your existing setup)
DB_CONFIG = {
    "user": "postgres",
    "password": "secret",
    "dbname": "mydb",
    "host": "localhost",
    "port": "5432",
    "autocommit": True
}

def get_dataset_info(hdf5_file):
    """Extract dataset information from HDF5 file."""
    info = {"keys": list(hdf5_file.keys())}
    
    # Get attributes
    distance = hdf5_file.attrs.get("distance", b"euclidean")
    if isinstance(distance, bytes):
        distance = distance.decode()
    info["distance"] = distance
    
    dtype = hdf5_file.attrs.get("type", b"dense")
    if isinstance(dtype, bytes):
        dtype = dtype.decode()
    info["type"] = dtype
    
    # Get label names if present
    if "label_names" in hdf5_file.attrs:
        label_names = hdf5_file.attrs["label_names"]
        info["label_names"] = [n.decode() if isinstance(n, bytes) else n for n in label_names]
    else:
        info["label_names"] = None
    
    # Determine vectors and dimension
    if "train_vec" in hdf5_file:
        info["train_key"] = "train_vec"
        info["train_shape"] = hdf5_file["train_vec"].shape
    elif "train" in hdf5_file:
        info["train_key"] = "train"
        info["train_shape"] = hdf5_file["train"].shape
    
    if "train_label" in hdf5_file:
        info["label_key"] = "train_label"
        info["label_shape"] = hdf5_file["train_label"].shape
    else:
        info["label_key"] = None
    
    if "test_vec" in hdf5_file:
        info["test_key"] = "test_vec"
    elif "test" in hdf5_file:
        info["test_key"] = "test"
    
    info["dimension"] = info["train_shape"][1]
    info["num_vectors"] = info["train_shape"][0]
    
    return info

def create_table(cur, table_name, dimension, label_names=None):
    """Create table for vector data."""
    cur.execute("DROP TABLE IF EXISTS " + table_name)
    
    if label_names:
        label_columns = ", ".join(name + " integer" for name in label_names)
        create_sql = "CREATE TABLE " + table_name + " (id integer, embedding vector(" + str(dimension) + "), " + label_columns + ")"
    else:
        create_sql = "CREATE TABLE " + table_name + " (id integer, embedding vector(" + str(dimension) + "))"
    
    cur.execute(create_sql)
    # Optimize storage for vectors
    #cur.execute("ALTER TABLE " + table_name + " ALTER COLUMN embedding SET STORAGE PLAIN")
    
    msg = "Created table  + table_name +  with vector(" + str(dimension) + ")"
    if label_names:
        msg += " and " + str(len(label_names)) + " label columns"
    print(msg)

def load_vectors_copy_binary(cur, table_name, vectors, labels=None, label_names=None, batch_size=100000):
    """Load vectors using efficient COPY BINARY (same as BigVectorBench)."""
    total = len(vectors)
    start_time = time.time()
    
    print("Loading " + str(total) + " vectors into  + table_name +  using COPY BINARY...")
    
    if labels is not None and label_names is not None:
        # With labels
        columns = "id, embedding, " + ", ".join(label_names)
        types = ["int4", "vector"] + ["int4" for _ in label_names]
        
        copy_sql = "COPY " + table_name + " (" + columns + ") FROM STDIN WITH (FORMAT BINARY)"
        with cur.copy(copy_sql) as copy:
            copy.set_types(types)
            for i, embedding in enumerate(vectors):
                row = (i, embedding.astype(np.float32)) + tuple(int(x) for x in labels[i])
                copy.write_row(row)
                
                if (i + 1) % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    pct = (i + 1) * 100 / total
                    print("  Loaded " + str(i+1) + "/" + str(total) + " (" + str(round(pct, 1)) + "%) - " + str(int(rate)) + " vec/s")
    else:
        # Without labels
        copy_sql = "COPY " + table_name + " (id, embedding) FROM STDIN WITH (FORMAT BINARY)"
        with cur.copy(copy_sql) as copy:
            copy.set_types(["int4", "vector"])
            for i, embedding in enumerate(vectors):
                copy.write_row((i, embedding.astype(np.float32)))
                
                if (i + 1) % batch_size == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    pct = (i + 1) * 100 / total
                    print("  Loaded " + str(i+1) + "/" + str(total) + " (" + str(round(pct, 1)) + "%) - " + str(int(rate)) + " vec/s")
    
    total_time = time.time() - start_time
    print("Completed loading " + str(total) + " vectors in " + str(round(total_time, 1)) + "s (" + str(int(total/total_time)) + " vec/s)")

def create_index(cur, table_name, index_type, metric="euclidean", m=16, ef_construction=200, lists=100):
    """Create vector index on the table."""
    index_name = table_name + "_embedding_idx"
    
    # Drop existing index
    cur.execute("DROP INDEX IF EXISTS " + index_name)
    
    # Choose operator based on metric
    if metric == "euclidean":
        ops = "vector_l2_ops"
    elif metric in ["angular", "cosine"]:
        ops = "vector_cosine_ops"
    else:
        ops = "vector_l2_ops"
    
    if index_type.lower() == "hnsw":
        create_idx_sql = "CREATE INDEX " + index_name + " ON " + table_name + " USING hnsw (embedding " + ops + ") WITH (m = " + str(m) + ", ef_construction = " + str(ef_construction) + ")"
    elif index_type.lower() == "ivfflat":
        create_idx_sql = "CREATE INDEX " + index_name + " ON " + table_name + " USING ivfflat (embedding " + ops + ") WITH (lists = " + str(lists) + ")"
    else:
        print("Unknown index type: " + index_type)
        return
    
    print("Creating " + index_type.upper() + " index (this may take a while)...")
    start_time = time.time()
    cur.execute(create_idx_sql)
    elapsed = time.time() - start_time
    print("Index created in " + str(round(elapsed, 1)) + "s")

def save_query_vectors(hdf5_file, info, output_file, num_queries=1000):
    """Save query vectors for workload execution."""
    test_key = info.get("test_key", "test")
    test_vectors = hdf5_file[test_key][:]
    
    # Sample query vectors
    if len(test_vectors) > num_queries:
        indices = np.random.choice(len(test_vectors), num_queries, replace=False)
        sampled = test_vectors[indices]
    else:
        sampled = test_vectors
        indices = np.arange(len(test_vectors))
    
    # Save as Python list (compatible with existing workload scripts)
    query_list = [vec.tolist() for vec in sampled]
    with open(output_file, "w") as f:
        f.write(repr(query_list))
    
    print("Saved " + str(len(query_list)) + " query vectors to  + output_file + ")
    
    # Save ground truth
    if "neighbors" in hdf5_file:
        neighbors = hdf5_file["neighbors"][:]
        if len(test_vectors) > num_queries:
            sampled_neighbors = neighbors[indices]
        else:
            sampled_neighbors = neighbors
        
        gt_file = output_file.replace(".txt", "_ground_truth.json")
        with open(gt_file, "w") as f:
            json.dump({
                "neighbors": sampled_neighbors.tolist(),
                "query_indices": indices.tolist()
            }, f)
        print("Saved ground truth to  + gt_file + ")

def main():
    parser = argparse.ArgumentParser(description="Load HDF5 vector dataset into PostgreSQL")
    parser.add_argument("--file", required=True, help="Path to HDF5 file")
    parser.add_argument("--table", default="items", help="Table name (default: items)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of vectors to load")
    parser.add_argument("--create-index", choices=["hnsw", "ivfflat", "none"], default="none",
                        help="Create index after loading")
    parser.add_argument("--m", type=int, default=16, help="HNSW M parameter")
    parser.add_argument("--ef-construction", type=int, default=200, help="HNSW ef_construction")
    parser.add_argument("--lists", type=int, default=100, help="IVFFlat lists parameter")
    parser.add_argument("--save-queries", type=str, default=None,
                        help="Save query vectors to file")
    parser.add_argument("--num-queries", type=int, default=1000, help="Number of query vectors")
    parser.add_argument("--info-only", action="store_true", help="Only print dataset info")
    
    args = parser.parse_args()
    
    print("Opening " + args.file + "...")
    with h5py.File(args.file, "r") as hdf5_file:
        info = get_dataset_info(hdf5_file)
        
        print("\nDataset Info:")
        print("  Type: " + str(info.get("type", "unknown")))
        print("  Distance: " + str(info.get("distance", "unknown")))
        print("  Dimension: " + str(info["dimension"]))
        print("  Num vectors: " + str(info["num_vectors"]))
        label_names = info.get("label_names")
        if label_names:
            print("  Label columns: " + ", ".join(label_names))
        else:
            print("  Label columns: none")
        
        if args.info_only:
            return
        
        # Save query vectors if requested
        if args.save_queries:
            save_query_vectors(hdf5_file, info, args.save_queries, args.num_queries)
        
        # Load vectors
        print("\nLoading vectors from  + info[train_key] + ...")
        vectors = hdf5_file[info["train_key"]][:]
        
        # Load labels if present
        labels = None
        if info.get("label_key") and label_names:
            print("Loading labels from  + info[label_key] + ...")
            labels = hdf5_file[info["label_key"]][:]
        
        # Apply limit
        if args.limit:
            vectors = vectors[:args.limit]
            if labels is not None:
                labels = labels[:args.limit]
            print("Limited to " + str(len(vectors)) + " vectors")
        
        # Connect to PostgreSQL
        print("\nConnecting to PostgreSQL...")
        conn = psycopg.connect(**DB_CONFIG)
        pgvector.psycopg.register_vector(conn)
        cur = conn.cursor()
        
        try:
            # Ensure vector extension exists
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create table
            create_table(cur, args.table, info["dimension"], label_names)
            
            # Load vectors
            load_vectors_copy_binary(cur, args.table, vectors, labels, label_names)
            
            # Create index if requested
            if args.create_index != "none":
                create_index(cur, args.table, args.create_index, 
                           info.get("distance", "euclidean"),
                           args.m, args.ef_construction, args.lists)
            
            # Print final stats
            cur.execute("SELECT COUNT(*) FROM " + args.table)
            count = cur.fetchone()[0]
            print("\nFinal count: " + str(count) + " vectors in  + args.table + ")
            
        finally:
            cur.close()
            conn.close()

if __name__ == "__main__":
    main()

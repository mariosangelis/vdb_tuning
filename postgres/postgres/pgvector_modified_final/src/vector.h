#ifndef VECTOR_H
#define VECTOR_H

#define VECTOR_MAX_DIM 16000

#define VECTOR_SIZE(_dim) (offsetof(Vector, x) + sizeof(float) * (_dim))
#define DatumGetVector(x) ((Vector *)PG_DETOAST_DATUM(x))
#define PG_GETARG_VECTOR_P(x) DatumGetVector(PG_GETARG_DATUM(x))
#define PG_RETURN_VECTOR_P(x) PG_RETURN_POINTER(x)

extern double read_buffer_extended_overhead;

extern double get_scan_lists_time;
extern double get_scan_items_time;
extern double read_buffer_extended_overhead;
extern double get_item_overhead;
extern double get_next_tuple_time;
extern double sort_overhead;
extern double copy_data_and_detach;
extern double create_workers;
extern double process_tuples_overhead;
extern double fetch_from_stream_time;
extern double calculate_distances_from_page_overhead;
extern double search_layer_overhead_hnsw;
extern double part1;
extern double part2;
extern int iteration;
extern long int calculations;

typedef struct Vector
{
	int32 vl_len_; /* varlena header (do not touch directly!) */
	int16 dim;	   /* number of dimensions */
	int16 unused;  /* reserved for future use, always zero */
	float x[FLEXIBLE_ARRAY_MEMBER];
} Vector;

Vector *InitVector(int dim);
void PrintVector(char *msg, Vector *vector);
int vector_cmp_internal(Vector *a, Vector *b);

/* TODO Move to better place */
#if PG_VERSION_NUM >= 160000
#define FUNCTION_PREFIX
#else
#define FUNCTION_PREFIX PGDLLEXPORT
#endif

#endif

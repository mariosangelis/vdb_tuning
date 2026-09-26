#include "postgres.h"

#include "miscadmin.h"			 // <-- needed for proc_exit
#include "postmaster/bgworker.h" // <-- needed for BackgroundWorker
#include "storage/dsm.h"
#include "storage/read_stream.h"
#include "storage/bufmgr.h"

#include "access/genam.h"

#include <float.h>
#include "nodes/pg_list.h"
#include "access/relscan.h"
#include "catalog/pg_operator_d.h"
#include "catalog/pg_type_d.h"
#include "lib/pairingheap.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "portability/instr_time.h"

#include "access/heapam.h"
#include "catalog/index.h"

#include <sys/mman.h> // for mmap, munmap, mincore, MAP_*
#include <unistd.h>	  // for sysconf()

#include "storage/procsignal.h"
#include "postmaster/bgworker.h"
#include "storage/ipc.h"

#define GetScanList(ptr) pairingheap_container(IvfflatScanList, ph_node, ptr)
#define GetScanListConst(ptr) pairingheap_const_container(IvfflatScanList, ph_node, ptr)

#define NUM_WORKERS 8

PGDLLEXPORT void ivf_prefetch_worker_main(Datum main_arg);

typedef struct BlockAccessEntry
{
	BlockNumber blocknum;
	int count;
} BlockAccessEntry;

static HTAB *centroid_access_counts = NULL;
static HTAB *page_hits = NULL;

static int number_of_scans = 0;
static int total_block_accesses = 0;
static int repeated_block_accesses = 0;

typedef struct ResultItem
{
	double distance;
	ItemPointerData tid;
} ResultItem;

typedef struct IvfParallelShared
{
	Oid relid;
	int num_blocks;				  // how many blocks for worker
	pg_atomic_uint32 worker_done; // 0 = not finished, 1 = finished
	int results_count;			  // how many results written
	int dimensions;
	Oid distfunc_oid;
	Oid collation;
	float4 scan_vector[1024]; // the distance scan value to match
	char data[FLEXIBLE_ARRAY_MEMBER];
} IvfParallelShared;

typedef struct IvfStaticScanState
{
	int index;
	int total;
	BlockNumber *blocknos;
} IvfStaticScanState;

typedef struct IvfflatMergeState
{
	int current[NUM_WORKERS];		  // Current index into each worker’s results
	int total[NUM_WORKERS];			  // Total results per worker
	ResultItem *results[NUM_WORKERS]; // Pointers to shareds[w]->results
} IvfflatMergeState;

IvfflatMergeState *merge;

int result_cmp(const void *a, const void *b)
{
	const ResultItem *ra = (const ResultItem *)a;
	const ResultItem *rb = (const ResultItem *)b;
	return (ra->distance > rb->distance) - (ra->distance < rb->distance);
}

void ivf_prefetch_worker_main(Datum main_arg)
{
	// elog(LOG, "[ivfflat_prefetch_worker] started");

	BackgroundWorkerUnblockSignals();
	BackgroundWorkerInitializeConnection("mydb", NULL, 0);

	dsm_handle handle = DatumGetUInt32(main_arg);
	dsm_segment *seg = dsm_attach(handle);

	IvfParallelShared *shared = dsm_segment_address(seg);
	Vector *vec = palloc0(VECTOR_SIZE(shared->dimensions));
	SET_VARSIZE(vec, VECTOR_SIZE(shared->dimensions));
	vec->dim = shared->dimensions;

	memcpy(vec->x, shared->scan_vector, sizeof(float4) * shared->dimensions);
	Datum value = PointerGetDatum(vec);

	StartTransactionCommand();
	Relation rel = relation_open(shared->relid, AccessShareLock);
	int result_index = 0;

	FmgrInfo procinfo;
	fmgr_info_cxt(shared->distfunc_oid, &procinfo, CurrentMemoryContext);
	BufferAccessStrategy bas = GetAccessStrategy(BAS_NORMAL);

	//elog(LOG, "Need to scan %d blocks", shared->num_blocks);
	BlockNumber *blocknos = (BlockNumber *)shared->data;
	ResultItem *results = (ResultItem *)(blocknos + shared->num_blocks);

	for (int i = 0; i < shared->num_blocks; i++)
	{
		BlockNumber blkno = blocknos[i];
		Buffer buf = ReadBufferExtended(rel, MAIN_FORKNUM, blkno, RBM_NORMAL, bas);

		LockBuffer(buf, BUFFER_LOCK_SHARE);

		Page page = BufferGetPage(buf);
		OffsetNumber maxoffno = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno++)
		{
			IndexTuple itup;
			Datum datum;
			bool isnull;

			ItemId itemid = PageGetItemId(page, offno);
			itup = (IndexTuple)PageGetItem(page, itemid);
			datum = index_getattr(itup, 1, RelationGetDescr(rel), &isnull);

			double dist = DatumGetFloat8(FunctionCall2Coll(&procinfo, shared->collation, datum, value));
			results[result_index].distance = dist;
			results[result_index].tid = itup->t_tid;
			result_index++;
		}
		UnlockReleaseBuffer(buf);
	}

	qsort(results, result_index, sizeof(ResultItem), result_cmp);
	//elog(LOG, "FINISHED, wrote %d results", result_index);

	shared->results_count = result_index;
	pg_atomic_write_u32(&shared->worker_done, 1);

	relation_close(rel, AccessShareLock);
	CommitTransactionCommand();

	proc_exit(0);
}

/*
 * Compare list distances
 */
static int
CompareLists(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (GetScanListConst(a)->distance > GetScanListConst(b)->distance)
		return 1;

	if (GetScanListConst(a)->distance < GetScanListConst(b)->distance)
		return -1;

	return 0;
}

IvfStaticScanState *scan_state;

typedef struct MetaBlockData
{
	int count;
	BlockNumber blocknos[FLEXIBLE_ARRAY_MEMBER]; // Flexible array member
} MetaBlockData;

/*
 * Get lists and sort by distance
 */
static void
GetScanLists(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque)scan->opaque;
	BlockNumber nextblkno = IVFFLAT_HEAD_BLKNO;
	int listCount = 0;
	double maxDistance = DBL_MAX;

	elog(INFO, "CALL GETSCANLISTS CALLED");

	/* Search all list pages */
	while (BlockNumberIsValid(nextblkno))
	{

		instr_time start, end, diff;
		instr_time start_overhead, end_overhead, elapsed;

		Buffer cbuf;
		Page cpage;
		OffsetNumber maxoffno;

		cbuf = ReadBuffer(scan->indexRelation, nextblkno);
		LockBuffer(cbuf, BUFFER_LOCK_SHARE);
		cpage = BufferGetPage(cbuf);

		maxoffno = PageGetMaxOffsetNumber(cpage);

		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			IvfflatList list = (IvfflatList)PageGetItem(cpage, PageGetItemId(cpage, offno));
			double distance;

			/* Use procinfo from the index instead of scan key for performance */
			distance = DatumGetFloat8(so->distfunc(so->procinfo, so->collation, PointerGetDatum(&list->center), value));

			if (listCount < so->maxProbes)
			{
				IvfflatScanList *scanlist;

				scanlist = &so->lists[listCount];
				scanlist->startPage = list->startPage;
				scanlist->numBlocks = list->numBlocks;

				scanlist->distance = distance;
				listCount++;

				/* Add to heap */
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Calculate max distance */
				if (listCount == so->maxProbes)
					maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
			else if (distance < maxDistance)
			{
				IvfflatScanList *scanlist;

				/* Remove */
				scanlist = GetScanList(pairingheap_remove_first(so->listQueue));

				/* Reuse */
				scanlist->startPage = list->startPage;
				scanlist->distance = distance;
				scanlist->numBlocks = list->numBlocks;
				pairingheap_add(so->listQueue, &scanlist->ph_node);

				/* Update max distance */
				maxDistance = GetScanList(pairingheap_first(so->listQueue))->distance;
			}
		}

		nextblkno = IvfflatPageGetOpaque(cpage)->nextblkno;

		UnlockReleaseBuffer(cbuf);
	}

	// for (int i = listCount - 1; i >= 0; i--)
	// {
	// 	IvfflatScanList *scanlist = GetScanList(pairingheap_remove_first(so->listQueue));
	// 	so->listPages[i] = scanlist->startPage;
	// }

	int num_blocks = 0;
	for (int i = listCount - 1; i >= 0; i--)
	{
		IvfflatScanList *scanlist = GetScanList(pairingheap_remove_first(so->listQueue));
		so->listPages[i] = scanlist->startPage;
		num_blocks += scanlist->numBlocks;
	}

	elog(INFO, "TOTAL blocks are %d\n", num_blocks);

	scan_state = palloc(sizeof(IvfStaticScanState));
	scan_state->index = 0;
	scan_state->total = 0;
	if (num_blocks > 0)
		scan_state->blocknos = palloc(sizeof(BlockNumber) * num_blocks);
	else
		scan_state->blocknos = NULL;

	for (int i = 0; i < listCount; i++)
	{
		BlockNumber searchPage = so->listPages[i];

		// Walk chain of metadata pages
		while (BlockNumberIsValid(searchPage))
		{
			Buffer metabuf;
			Page metapage;

			metabuf = ReadBufferExtended(scan->indexRelation, MAIN_FORKNUM, searchPage, RBM_NORMAL, so->bas);
			LockBuffer(metabuf, BUFFER_LOCK_SHARE);
			metapage = BufferGetPage(metabuf);

			ItemId itemid = PageGetItemId(metapage, FirstOffsetNumber);
			MetaBlockData *metaDataItem = (MetaBlockData *)PageGetItem(metapage, itemid);

			// elog(INFO, "count is %d", metaDataItem->count);

			for (int j = 0; j < metaDataItem->count; j++)
			{
				scan_state->blocknos[scan_state->total++] = metaDataItem->blocknos[j];
			}
			searchPage = IvfflatPageGetOpaque(metapage)->nextblkno;
			UnlockReleaseBuffer(metabuf);
		}
	}

	Assert(pairingheap_is_empty(so->listQueue));
}

/*
 * Get items
 */

static void
GetScanItems(IndexScanDesc scan, Datum value)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque)scan->opaque;
	TupleDesc tupdesc = RelationGetDescr(scan->indexRelation);
	TupleTableSlot *slot = so->vslot;
	int batchProbes = 0;

	tuplesort_reset(so->sortstate);
	instr_time start_overhead, end_overhead, elapsed;
	process_tuples_overhead = 0.0;
	fetch_from_stream_time = 0.0;

	int chunk_size = (scan_state->total + NUM_WORKERS - 1) / NUM_WORKERS;

	dsm_segment *segments[NUM_WORKERS];
	IvfParallelShared *shareds[NUM_WORKERS];
	BackgroundWorkerHandle *handles[NUM_WORKERS];

	Vector *vec = DatumGetVector(value);

	for (int w = 0; w < NUM_WORKERS; w++)
	{
		int start = w * chunk_size;
		int end = Min(start + chunk_size, scan_state->total);
		int count = end - start;
		int max_results = count * ((BLCKSZ - 24) / (vec->dim * sizeof(float)));
		elog(INFO, "MAX RESULT SIZE for worker IS  %d", max_results);

		// Size sz = offsetof(IvfParallelShared, blocknos)  + sizeof(BlockNumber) * count;
		Size sz = offsetof(IvfParallelShared, data) + sizeof(BlockNumber) * count + sizeof(ResultItem) * max_results;
		segments[w] = dsm_create(sz, 0);
		shareds[w] = dsm_segment_address(segments[w]);

		shareds[w]->relid = RelationGetRelid(scan->indexRelation);
		shareds[w]->num_blocks = count;
		pg_atomic_init_u32(&shareds[w]->worker_done, 0);
		memcpy(shareds[w]->data, &scan_state->blocknos[start], sizeof(BlockNumber) * count);
		shareds[w]->results_count = 0;
		shareds[w]->dimensions = vec->dim;
		memcpy(shareds[w]->scan_vector, vec->x, sizeof(float4) * vec->dim);
		shareds[w]->distfunc_oid = so->procinfo->fn_oid;
		shareds[w]->collation = so->collation;

		// Launch worker
		BackgroundWorker worker;
		memset(&worker, 0, sizeof(worker));
		worker.bgw_flags = BGWORKER_SHMEM_ACCESS | BGWORKER_BACKEND_DATABASE_CONNECTION;
		worker.bgw_start_time = BgWorkerStart_ConsistentState;
		worker.bgw_restart_time = BGW_NEVER_RESTART;
		snprintf(worker.bgw_name, BGW_MAXLEN, "ivfflat worker %d", w);
		snprintf(worker.bgw_library_name, BGW_MAXLEN, "vector");
		snprintf(worker.bgw_function_name, BGW_MAXLEN, "ivf_prefetch_worker_main");
		worker.bgw_main_arg = UInt32GetDatum(dsm_segment_handle(segments[w]));

		if (!RegisterDynamicBackgroundWorker(&worker, &handles[w]))
			elog(INFO, "could not register worker %d", w);
	}

	bool all_done = false;
	while (!all_done)
	{
		all_done = true;
		for (int w = 0; w < NUM_WORKERS; w++)
		{
			if (pg_atomic_read_u32(&shareds[w]->worker_done) == 0)
			{
				all_done = false;
				break;
			}
		}
		if (!all_done)
			pg_usleep(1000);
	}

	merge = palloc(sizeof(IvfflatMergeState));

	for (int w = 0; w < NUM_WORKERS; w++)
	{
		BlockNumber *blocknos = (BlockNumber *)shareds[w]->data;
		ResultItem *results = (ResultItem *)(blocknos + shareds[w]->num_blocks);
		merge->current[w] = 0;
		merge->total[w] = shareds[w]->results_count;
		merge->results[w] = palloc(sizeof(ResultItem) * merge->total[w]);
		memcpy(merge->results[w], results, sizeof(ResultItem) * merge->total[w]);
	}

	for (int w = 0; w < NUM_WORKERS; w++)
	{
		dsm_detach(segments[w]);
	}

	tuplesort_performsort(so->sortstate);
#if defined(IVFFLAT_MEMORY)
	elog(INFO, "memory: %zu MB", MemoryContextMemAllocated(CurrentMemoryContext, true) / (1024 * 1024));
#endif
}

/*
 * Zero distance
 */
static Datum
ZeroDistance(FmgrInfo *flinfo, Oid collation, Datum arg1, Datum arg2)
{
	return Float8GetDatum(0.0);
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{

	IvfflatScanOpaque so = (IvfflatScanOpaque)scan->opaque;
	Datum value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
	{
		value = PointerGetDatum(NULL);
		so->distfunc = ZeroDistance;
	}
	else
	{
		value = scan->orderByData->sk_argument;
		so->distfunc = FunctionCall2Coll;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->normprocinfo != NULL)
		{
			MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);

			value = IvfflatNormValue(so->typeInfo, so->collation, value);

			MemoryContextSwitchTo(oldCtx);
		}
	}

	return value;
}

/*
 * Initialize scan sort state
 */
static Tuplesortstate *
InitScanSortState(TupleDesc tupdesc)
{
	AttrNumber attNums[] = {1};
	Oid sortOperators[] = {Float8LessOperator};
	Oid sortCollations[] = {InvalidOid};
	bool nullsFirstFlags[] = {false};

	return tuplesort_begin_heap(tupdesc, 1, attNums, sortOperators, sortCollations, nullsFirstFlags, work_mem, NULL, false);
}

/*
 * Prepare for an index scan
 */
IndexScanDesc
ivfflatbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	IvfflatScanOpaque so;
	int lists;
	int dimensions;
	int probes = ivfflat_probes;
	int maxProbes;
	MemoryContext oldCtx;

	// init_num_workers();

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	/* Get lists and dimensions from metapage */
	IvfflatGetMetaPageInfo(index, &lists, &dimensions);

	if (ivfflat_iterative_scan != IVFFLAT_ITERATIVE_SCAN_OFF)
		maxProbes = Max(ivfflat_max_probes, probes);
	else
		maxProbes = probes;

	if (probes > lists)
		probes = lists;

	if (maxProbes > lists)
		maxProbes = lists;

	so = (IvfflatScanOpaque)palloc(sizeof(IvfflatScanOpaqueData));
	so->typeInfo = IvfflatGetTypeInfo(index);
	so->first = true;
	so->probes = probes;
	so->maxProbes = maxProbes;
	so->dimensions = dimensions;

	/* Set support functions */
	so->procinfo = index_getprocinfo(index, 1, IVFFLAT_DISTANCE_PROC);
	so->normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	so->collation = index->rd_indcollation[0];

	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "Ivfflat scan temporary context",
									   ALLOCSET_DEFAULT_SIZES);

	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/* Create tuple description for sorting */
	so->tupdesc = CreateTemplateTupleDesc(2);
	TupleDescInitEntry(so->tupdesc, (AttrNumber)1, "distance", FLOAT8OID, -1, 0);
	TupleDescInitEntry(so->tupdesc, (AttrNumber)2, "heaptid", TIDOID, -1, 0);

	/* Prep sort */
	so->sortstate = InitScanSortState(so->tupdesc);

	/* Need separate slots for puttuple and gettuple */
	so->vslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsVirtual);
	so->mslot = MakeSingleTupleTableSlot(so->tupdesc, &TTSOpsMinimalTuple);

	/*
	 * Reuse same set of shared buffers for scan
	 *
	 * See postgres/src/backend/storage/buffer/README for description
	 */
	so->bas = GetAccessStrategy(BAS_NORMAL);

	so->listQueue = pairingheap_allocate(CompareLists, scan);
	so->listPages = palloc(maxProbes * sizeof(BlockNumber));
	so->listIndex = 0;
	so->lists = palloc(maxProbes * sizeof(IvfflatScanList));

	MemoryContextSwitchTo(oldCtx);

	scan->opaque = so;

	return scan;
}

/*
 * Start or restart an index scan
 */
void ivfflatrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque)scan->opaque;

	so->first = true;
	pairingheap_reset(so->listQueue);
	so->listIndex = 0;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

/*
 * Fetch the next tuple in the given scan
 */
bool ivfflatgettuple(IndexScanDesc scan, ScanDirection dir)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque)scan->opaque;
	ItemPointer heaptid;
	bool isnull;

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{

		Datum value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan ivfflat index without order");

		/* Requires MVCC-compliant snapshot as not able to pin during sorting */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with ivfflat");

		value = GetScanValue(scan);

		struct timespec start, end;
		double elapsed;

		GetScanLists(scan, value);
		GetScanItems(scan, value);

		so->first = false;
		so->value = value;
	}
	struct timespec start, end;
	double elapsed;

	// Measure remaining time
	// clock_gettime(CLOCK_MONOTONIC, &start);
	// while (!tuplesort_gettupleslot(so->sortstate, true, false, so->mslot, NULL))
	// {
	// 	if (so->listIndex == so->maxProbes)
	// 		return false;

	// 	IvfflatBench("GetScanItems", GetScanItems(scan, so->value));
	// }

	int best_worker = -1;
	double best_distance = DBL_MAX;

	for (int w = 0; w < NUM_WORKERS; w++)
	{
		if (merge->current[w] < merge->total[w])
		{
			double dist = merge->results[w][merge->current[w]].distance;
			if (dist < best_distance)
			{
				best_distance = dist;
				best_worker = w;
			}
		}
	}

	ResultItem *item = &merge->results[best_worker][merge->current[best_worker]++];

	// Datum distDatum = slot_getattr(so->mslot, 1, &isnull);
	//	elog(INFO, "📦 return item = (%u, %u)", ItemPointerGetBlockNumber(&item->tid), ItemPointerGetOffsetNumber(&item->tid));
	// heaptid = (ItemPointer)DatumGetPointer(slot_getattr(so->mslot, 2, &isnull));
	//        elog(INFO, "📦 from slots return item = %d", *heapid);
	//	elog(INFO, "📦 return HEAP TID: block = %u, offset = %u",     ItemPointerGetBlockNumber(heaptid), ItemPointerGetOffsetNumber(heaptid));

	// BlockAccessEntry *page_map_entry;
	// bool page_found;

	// BlockNumber page_block = ItemPointerGetBlockNumber(heaptid);
	// page_map_entry = hash_search(page_hits, &page_block, HASH_ENTER, &page_found);

	// scan->xs_heaptid = *heaptid;
	scan->xs_heaptid = item->tid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;

	scan->xs_orderbyvals = palloc(sizeof(Datum) * scan->numberOfOrderBys);
	scan->xs_orderbynulls = palloc(sizeof(bool) * scan->numberOfOrderBys);

	// scan->xs_orderbyvals[0] = distDatum; // Float8 datum
	scan->xs_orderbyvals[0] = Float8GetDatum(item->distance);
	scan->xs_orderbynulls[0] = false;

	return true;
}

/*
 * End a scan and release resources
 */
void ivfflatendscan(IndexScanDesc scan)
{
	IvfflatScanOpaque so = (IvfflatScanOpaque)scan->opaque;

	/* Free any temporary files */
	tuplesort_end(so->sortstate);

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}

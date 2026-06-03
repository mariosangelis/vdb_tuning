#include "postgres.h"

#include <float.h>
#include <math.h>
#include <omp.h>

#include "bitvec.h"
#include "halfutils.h"
#include "halfvec.h"
#include "ivfflat.h"
#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "vector.h"

/*
 * Inline distance: computes same result as the IVFFLAT_KMEANS_DISTANCE_PROC
 * for L2 (Euclidean) distance. For cosine/IP it uses angular distance.
 * Thread-safe: no palloc.
 */
static inline float
inline_vector_distance(const float *ax, const float *bx, int dim, bool use_angular)
{
	if (!use_angular)
	{
		/* L2 distance = sqrt(sum of squared differences) */
		float distance = 0.0f;
		for (int i = 0; i < dim; i++)
		{
			float diff = ax[i] - bx[i];
			distance += diff * diff;
		}
		return sqrtf(distance);
	}
	else
	{
		/* Angular distance for cosine/IP */
		float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
		for (int i = 0; i < dim; i++)
		{
			dot += ax[i] * bx[i];
			norm_a += ax[i] * ax[i];
			norm_b += bx[i] * bx[i];
		}
		float denom = sqrtf(norm_a) * sqrtf(norm_b);
		if (denom == 0.0f)
			return 1.0f;
		float sim = dot / denom;
		if (sim > 1.0f) sim = 1.0f;
		if (sim < -1.0f) sim = -1.0f;
		return acosf(sim) / M_PI;
	}
}

/*
 * Get the float* pointer to vector data from a VectorArray entry
 */
static inline float *
get_vector_data(VectorArray arr, int offset)
{
	Vector *v = (Vector *) VectorArrayGet(arr, offset);
	return v->x;
}

/*
 * Initialize with kmeans++
 *
 * https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf
 */
static void
InitCenters(Relation index, VectorArray samples, VectorArray centers, float *lowerBound, bool use_angular)
{
	int64		j;
	int			numCenters = centers->maxlen;
	int			numSamples = samples->length;
	int			dim = samples->dim;
	float	   *weight = palloc(numSamples * sizeof(float));

	/* Choose an initial center uniformly at random */
	VectorArraySet(centers, 0, VectorArrayGet(samples, RandomInt() % numSamples));
	centers->length++;

	for (j = 0; j < numSamples; j++)
		weight[j] = FLT_MAX;

	elog(INFO, "k-means++ init: seeding %d centers from %d samples (%d threads)",
		 numCenters, numSamples, omp_get_max_threads());

	for (int i = 0; i < numCenters; i++)
	{
		double		sum;
		double		choice;

		CHECK_FOR_INTERRUPTS();

		if (i % 1000 == 0)
			elog(INFO, "k-means++ init: center %d / %d", i, numCenters);

		float *center_i_data = get_vector_data(centers, i);

		/* Parallel: compute distance from each sample to center i */
		#pragma omp parallel for schedule(static)
		for (int64 jj = 0; jj < numSamples; jj++)
		{
			float *sample_data = get_vector_data(samples, jj);
			float distance = inline_vector_distance(sample_data, center_i_data, dim, use_angular);

			lowerBound[jj * numCenters + i] = distance;

			float dist_sq = distance * distance;
			if (dist_sq < weight[jj])
				weight[jj] = dist_sq;
		}

		/* Only compute lower bound on last iteration */
		if (i + 1 == numCenters)
			break;

		/* Sum weights (sequential — fast, just numSamples additions) */
		sum = 0.0;
		for (j = 0; j < numSamples; j++)
			sum += weight[j];

		/* Choose new center using weighted probability distribution. */
		choice = sum * RandomDouble();
		for (j = 0; j < numSamples - 1; j++)
		{
			choice -= weight[j];
			if (choice <= 0)
				break;
		}

		VectorArraySet(centers, i + 1, VectorArrayGet(samples, j));
		centers->length++;
	}

	pfree(weight);
	elog(INFO, "k-means++ initialization complete");
}

/*
 * Norm centers
 */
static void
NormCenters(const IvfflatTypeInfo * typeInfo, Oid collation, VectorArray centers)
{
	MemoryContext normCtx = AllocSetContextCreate(CurrentMemoryContext,
												  "Ivfflat norm temporary context",
												  ALLOCSET_DEFAULT_SIZES);
	MemoryContext oldCtx = MemoryContextSwitchTo(normCtx);

	for (int j = 0; j < centers->length; j++)
	{
		Datum		center = PointerGetDatum(VectorArrayGet(centers, j));
		Datum		newCenter = IvfflatNormValue(typeInfo, collation, center);
		Size		size = VARSIZE_ANY(DatumGetPointer(newCenter));

		if (size > centers->itemsize)
			elog(ERROR, "safety check failed");

		memcpy(DatumGetPointer(center), DatumGetPointer(newCenter), size);
		MemoryContextReset(normCtx);
	}

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(normCtx);
}

/*
 * Quick approach if we have no data
 */
static void
RandomCenters(Relation index, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	int			dimensions = centers->dim;
	FmgrInfo   *normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
	Oid			collation = index->rd_indcollation[0];
	float	   *x = (float *) palloc(sizeof(float) * dimensions);

	/* Fill with random data */
	while (centers->length < centers->maxlen)
	{
		Pointer		center = VectorArrayGet(centers, centers->length);

		for (int i = 0; i < dimensions; i++)
			x[i] = (float) RandomDouble();

		typeInfo->updateCenter(center, dimensions, x);

		centers->length++;
	}

	if (normprocinfo != NULL)
		NormCenters(typeInfo, collation, centers);
}

#ifdef IVFFLAT_MEMORY
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(MemoryContext context, Size estimatedSize)
{
	elog(INFO, "total memory: %zu MB",
		 MemoryContextMemAllocated(context, true) / (1024 * 1024));
	elog(INFO, "estimated memory: %zu MB", estimatedSize / (1024 * 1024));
}
#endif

/*
 * Sum centers
 */
static void
SumCenters(VectorArray samples, float *agg, int *closestCenters, const IvfflatTypeInfo * typeInfo)
{
	for (int j = 0; j < samples->length; j++)
	{
		float	   *x = agg + ((int64) closestCenters[j] * samples->dim);

		typeInfo->sumCenter(VectorArrayGet(samples, j), x);
	}
}

/*
 * Update centers
 */
static void
UpdateCenters(float *agg, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	for (int j = 0; j < centers->length; j++)
	{
		float	   *x = agg + ((int64) j * centers->dim);

		typeInfo->updateCenter(VectorArrayGet(centers, j), centers->dim, x);
	}
}

/*
 * Compute new centers — UNCHANGED from original
 */
static void
ComputeNewCenters(VectorArray samples, float *agg, VectorArray newCenters, int *centerCounts, int *closestCenters, FmgrInfo *normprocinfo, Oid collation, const IvfflatTypeInfo * typeInfo)
{
	int			dimensions = newCenters->dim;
	int			numCenters = newCenters->length;
	int			numSamples = samples->length;

	/* Reset sum and count */
	for (int j = 0; j < numCenters; j++)
	{
		float	   *x = agg + ((int64) j * dimensions);

		for (int k = 0; k < dimensions; k++)
			x[k] = 0.0;

		centerCounts[j] = 0;
	}

	/* Increment sum of closest center */
	SumCenters(samples, agg, closestCenters, typeInfo);

	/* Increment count of closest center */
	for (int j = 0; j < numSamples; j++)
		centerCounts[closestCenters[j]] += 1;

	/* Divide sum by count */
	for (int j = 0; j < numCenters; j++)
	{
		float	   *x = agg + ((int64) j * dimensions);

		if (centerCounts[j] > 0)
		{
			/* Double avoids overflow, but requires more memory */
			/* TODO Update bounds */
			for (int k = 0; k < dimensions; k++)
			{
				if (isinf(x[k]))
					x[k] = x[k] > 0 ? FLT_MAX : -FLT_MAX;
			}

			for (int k = 0; k < dimensions; k++)
				x[k] /= centerCounts[j];
		}
		else
		{
			/* TODO Handle empty centers properly */
			for (int k = 0; k < dimensions; k++)
				x[k] = RandomDouble();
		}
	}

	/* Set new centers */
	UpdateCenters(agg, newCenters, typeInfo);

	/* Normalize if needed */
	if (normprocinfo != NULL)
		NormCenters(typeInfo, collation, newCenters);
}

/*
 * Use Elkan for performance. This requires distance function to satisfy triangle inequality.
 *
 * We use L2 distance for L2 (not L2 squared like index scan)
 * and angular distance for inner product and cosine distance
 *
 * https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf
 *
 * PARALLELIZED with OpenMP: Steps 1, 2-3, 5, 6 run in parallel.
 * Step 4 (ComputeNewCenters) uses original sequential code (uses palloc/typeInfo).
 */
static void
ElkanKmeans(Relation index, VectorArray samples, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
	int			dimensions = centers->dim;
	int			numCenters = centers->maxlen;
	int			numSamples = samples->length;
	VectorArray newCenters;
	float	   *agg;
	int		   *centerCounts;
	int		   *closestCenters;
	float	   *lowerBound;
	float	   *upperBound;
	float	   *s;
	float	   *halfcdist;
	float	   *newcdist;
	bool		use_angular;

	/* Calculate allocation sizes */
	Size		samplesSize = VECTOR_ARRAY_SIZE(samples->maxlen, samples->itemsize);
	Size		centersSize = VECTOR_ARRAY_SIZE(centers->maxlen, centers->itemsize);
	Size		newCentersSize = VECTOR_ARRAY_SIZE(numCenters, centers->itemsize);
	Size		aggSize = sizeof(float) * (int64) numCenters * dimensions;
	Size		centerCountsSize = sizeof(int) * numCenters;
	Size		closestCentersSize = sizeof(int) * numSamples;
	Size		lowerBoundSize = sizeof(float) * numSamples * numCenters;
	Size		upperBoundSize = sizeof(float) * numSamples;
	Size		sSize = sizeof(float) * numCenters;
	Size		halfcdistSize = sizeof(float) * numCenters * numCenters;
	Size		newcdistSize = sizeof(float) * numCenters;

	/* Calculate total size */
	Size		totalSize = samplesSize + centersSize + newCentersSize + aggSize + centerCountsSize + closestCentersSize + lowerBoundSize + upperBoundSize + sSize + halfcdistSize + newcdistSize;

	/* Check memory requirements */
	/* Add one to error message to ceil */
	if (totalSize > (Size) maintenance_work_mem * 1024L)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("memory required is %zu MB, maintenance_work_mem is %d MB",
						totalSize / (1024 * 1024) + 1, maintenance_work_mem / 1024)));

	/* Ensure indexing does not overflow */
	if (numCenters * numCenters > INT_MAX)
		elog(ERROR, "Indexing overflow detected. Please report a bug.");

	/* Set support functions */
	procinfo = index_getprocinfo(index, 1, IVFFLAT_KMEANS_DISTANCE_PROC);
	normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_KMEANS_NORM_PROC);
	collation = index->rd_indcollation[0];

	/* Determine distance type: if norm proc exists, it's cosine/IP (angular) */
	use_angular = (normprocinfo != NULL);

	/* Allocate space */
	/* Use float instead of double to save memory */
	agg = palloc(aggSize);
	centerCounts = palloc(centerCountsSize);
	closestCenters = palloc(closestCentersSize);
	lowerBound = palloc_extended(lowerBoundSize, MCXT_ALLOC_HUGE);
	upperBound = palloc(upperBoundSize);
	s = palloc(sSize);
	halfcdist = palloc_extended(halfcdistSize, MCXT_ALLOC_HUGE);
	newcdist = palloc(newcdistSize);

	/* Initialize new centers */
	newCenters = VectorArrayInit(numCenters, dimensions, centers->itemsize);
	newCenters->length = numCenters;

#ifdef IVFFLAT_MEMORY
	ShowMemoryUsage(MemoryContextGetParent(CurrentMemoryContext), totalSize);
#endif

	elog(INFO, "Parallel k-means: %d samples, %d centers, %d dims, %d threads",
		 numSamples, numCenters, dimensions, omp_get_max_threads());

	/* Pick initial centers */
	InitCenters(index, samples, centers, lowerBound, use_angular);

	/* Assign each x to its closest initial center c(x) = argmin d(x,c) */
	#pragma omp parallel for schedule(static)
	for (int64 j = 0; j < numSamples; j++)
	{
		float		minDistance = FLT_MAX;
		int			closestCenter = 0;

		/* Find closest center */
		for (int64 k = 0; k < numCenters; k++)
		{
			/* TODO Use Lemma 1 in k-means++ initialization */
			float		distance = lowerBound[j * numCenters + k];

			if (distance < minDistance)
			{
				minDistance = distance;
				closestCenter = k;
			}
		}

		upperBound[j] = minDistance;
		closestCenters[j] = closestCenter;
	}

	elog(INFO, "Starting Elkan iterations...");

	/* Give 500 iterations to converge */
	for (int iteration = 0; iteration < 500; iteration++)
	{
		int			changes = 0;
		bool		rjreset;

		/* Can take a while, so ensure we can interrupt */
		CHECK_FOR_INTERRUPTS();

		/* Step 1: For all centers, compute pairwise distance — PARALLEL */
		#pragma omp parallel for schedule(dynamic)
		for (int64 j = 0; j < numCenters; j++)
		{
			float *cj_data = get_vector_data(centers, j);

			for (int64 k = j + 1; k < numCenters; k++)
			{
				float *ck_data = get_vector_data(centers, k);
				float distance = 0.5f * inline_vector_distance(cj_data, ck_data, dimensions, use_angular);

				halfcdist[j * numCenters + k] = distance;
				halfcdist[k * numCenters + j] = distance;
			}
		}

		/* For all centers c, compute s(c) — PARALLEL */
		#pragma omp parallel for schedule(static)
		for (int64 j = 0; j < numCenters; j++)
		{
			float		minDistance = FLT_MAX;

			for (int64 k = 0; k < numCenters; k++)
			{
				float		distance;

				if (j == k)
					continue;

				distance = halfcdist[j * numCenters + k];
				if (distance < minDistance)
					minDistance = distance;
			}

			s[j] = minDistance;
		}

		rjreset = iteration != 0;

		/* Steps 2-3: Assignment update — PARALLEL (biggest hotspot) */
		#pragma omp parallel for schedule(dynamic, 256) reduction(+:changes)
		for (int64 j = 0; j < numSamples; j++)
		{
			bool		rj;

			/* Step 2: Identify all points x such that u(x) <= s(c(x)) */
			if (upperBound[j] <= s[closestCenters[j]])
				continue;

			rj = rjreset;

			for (int64 k = 0; k < numCenters; k++)
			{
				float		dxcx;

				/* Step 3: For all remaining points x and centers c */
				if (k == closestCenters[j])
					continue;

				if (upperBound[j] <= lowerBound[j * numCenters + k])
					continue;

				if (upperBound[j] <= halfcdist[closestCenters[j] * numCenters + k])
					continue;

				float *sample_data = get_vector_data(samples, j);

				/* Step 3a */
				if (rj)
				{
					float *cc_data = get_vector_data(centers, closestCenters[j]);
					dxcx = inline_vector_distance(sample_data, cc_data, dimensions, use_angular);

					/* d(x,c(x)) computed, which is a form of d(x,c) */
					lowerBound[j * numCenters + closestCenters[j]] = dxcx;
					upperBound[j] = dxcx;

					rj = false;
				}
				else
					dxcx = upperBound[j];

				/* Step 3b */
				if (dxcx > lowerBound[j * numCenters + k] || dxcx > halfcdist[closestCenters[j] * numCenters + k])
				{
					float *ck_data = get_vector_data(centers, k);
					float dxc = inline_vector_distance(sample_data, ck_data, dimensions, use_angular);

					/* d(x,c) calculated */
					lowerBound[j * numCenters + k] = dxc;

					if (dxc < dxcx)
					{
						closestCenters[j] = k;

						/* c(x) changed */
						upperBound[j] = dxc;

						changes++;
					}
				}
			}
		}

		/* Step 4: For each center c, let m(c) be mean of all points assigned */
		/* This uses original sequential code with typeInfo/palloc — unchanged */
		ComputeNewCenters(samples, agg, newCenters, centerCounts, closestCenters, normprocinfo, collation, typeInfo);

		/* Step 5: compute center movement — PARALLEL */
		#pragma omp parallel for schedule(static)
		for (int j = 0; j < numCenters; j++)
		{
			float *old_data = get_vector_data(centers, j);
			float *new_data = get_vector_data(newCenters, j);
			newcdist[j] = inline_vector_distance(old_data, new_data, dimensions, use_angular);
		}

		/* Step 5b: update lower bounds — PARALLEL */
		#pragma omp parallel for schedule(static)
		for (int64 j = 0; j < numSamples; j++)
		{
			for (int64 k = 0; k < numCenters; k++)
			{
				float		distance = lowerBound[j * numCenters + k] - newcdist[k];

				if (distance < 0)
					distance = 0;

				lowerBound[j * numCenters + k] = distance;
			}
		}

		/* Step 6: update upper bounds — PARALLEL */
		#pragma omp parallel for schedule(static)
		for (int j = 0; j < numSamples; j++)
			upperBound[j] += newcdist[closestCenters[j]];

		/* Step 7 */
		for (int j = 0; j < numCenters; j++)
			VectorArraySet(centers, j, VectorArrayGet(newCenters, j));

		if (iteration > 0 && iteration % 10 == 0)
			elog(INFO, "k-means iteration %d: %d changes", iteration, changes);

		if (changes == 0 && iteration != 0)
		{
			elog(INFO, "k-means converged after %d iterations", iteration + 1);
			break;
		}
	}
}

/*
 * Ensure no NaN or infinite values
 */
static void
CheckElements(VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	float	   *scratch = palloc(sizeof(float) * centers->dim);

	for (int i = 0; i < centers->length; i++)
	{
		for (int j = 0; j < centers->dim; j++)
			scratch[j] = 0;

		/* /fp:fast may not propagate NaN with MSVC, but that's alright */
		typeInfo->sumCenter(VectorArrayGet(centers, i), scratch);

		for (int j = 0; j < centers->dim; j++)
		{
			if (isnan(scratch[j]))
				elog(ERROR, "NaN detected. Please report a bug.");

			if (isinf(scratch[j]))
				elog(ERROR, "Infinite value detected. Please report a bug.");
		}
	}
}

/*
 * Ensure no zero vectors for cosine distance
 */
static void
CheckNorms(VectorArray centers, Relation index)
{
	/* Check NORM_PROC instead of KMEANS_NORM_PROC */
	FmgrInfo   *normprocinfo = IvfflatOptionalProcInfo(index, IVFFLAT_NORM_PROC);
	Oid			collation = index->rd_indcollation[0];

	if (normprocinfo == NULL)
		return;

	for (int i = 0; i < centers->length; i++)
	{
		double		norm = DatumGetFloat8(FunctionCall1Coll(normprocinfo, collation, PointerGetDatum(VectorArrayGet(centers, i))));

		if (norm == 0)
			elog(ERROR, "Zero norm detected. Please report a bug.");
	}
}

/*
 * Detect issues with centers
 */
static void
CheckCenters(Relation index, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	if (centers->length != centers->maxlen)
		elog(ERROR, "Not enough centers. Please report a bug.");

	CheckElements(centers, typeInfo);
	CheckNorms(centers, index);
}

/*
 * Perform naive k-means centering
 * We use spherical k-means for inner product and cosine
 */
void
IvfflatKmeans(Relation index, VectorArray samples, VectorArray centers, const IvfflatTypeInfo * typeInfo)
{
	MemoryContext kmeansCtx = AllocSetContextCreate(CurrentMemoryContext,
													"Ivfflat kmeans temporary context",
													ALLOCSET_DEFAULT_SIZES);
	MemoryContext oldCtx = MemoryContextSwitchTo(kmeansCtx);

	if (samples->length == 0)
		RandomCenters(index, centers, typeInfo);
	else
		ElkanKmeans(index, samples, centers, typeInfo);

	CheckCenters(index, centers, typeInfo);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(kmeansCtx);
}

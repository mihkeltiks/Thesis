#ifndef CLI_H
#define CLI_H

// CLI argument parsing for the benchmarks. Header-only (static functions).
// Inspired by TheBandwidthBenchmark/src/cli.c / cli.h.

#include <ctype.h>
#include <errno.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define HLINE "------------------------------------------------------------------------\n"

// ---------------------------------------------------------------------------
// Compile-time defaults (overridden by -DSIZE=... etc. in the mk files)
// ---------------------------------------------------------------------------
#ifdef SIZE
#  define DEFAULT_SIZE        ((size_t)(SIZE))
#else
#  define DEFAULT_SIZE        ((size_t)5120)
#endif

#ifdef NTIMES
#  define DEFAULT_NTIMES      NTIMES
#else
#  define DEFAULT_NTIMES      500
#endif

// ---------------------------------------------------------------------------
// Argument struct
// ---------------------------------------------------------------------------
typedef enum {
    INIT_RANDOM   = 0,  // fill matrices with curand uniform random values
    INIT_CONSTANT = 1   // fill matrices with a fixed constant value
} InitMode;

typedef struct {
    int      device_id;       // GPU device index
    size_t   matrix_size;     // matrix dimension
    int      repeats;         // iterations per run
    InitMode init_mode;       // data initialisation: random or constant
} GemmArgs;

// ---------------------------------------------------------------------------
// Help text
// ---------------------------------------------------------------------------
static void printHelp(const char *prog)
{
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -h              Show this help text\n");
    printf("  -d <int>        GPU device ID (default: 0)\n");
    printf("  -s <int>        Matrix size (default: %zu)\n", DEFAULT_SIZE);
    printf("  -n <int>        Number of iterations (default: %d)\n", DEFAULT_NTIMES);
    printf("  -i <mode>       Data initialisation mode: 'random' or 'constant' (default: random)\n");
    printf("\nExamples:\n");
    printf("  %s -d 0 -s 16384 -n 500\n", prog);
#if defined(GEMM_SLEEP)
    printf("\ngemm_sleep (host sleeps between GEMM iterations):\n");
    printf("  -m <sec>       Sleep seconds per lap (default 1.0; 0 = GEMM only)\n");
#elif defined(GEMM_COMMS) && defined(GEMM_COMMS_NCCL)
    printf("\ngemm_comms_nccl (MPI rank setup + NCCL collectives):\n");
    printf("  -m <size>        Collective buffer size; accepts K/M/G suffix (e.g. 32M, 1G).\n");
    printf("                   Default: COMM_MSG_SIZE or 32M. 0 disables collectives.\n");
    printf("  -p allreduce|allgather|reducescatter|ce_allgather\n");
    printf("                   Communication pattern.\n");
    printf("  --barrier       Rank sync each lap (every lap starts simultaneously on all\n");
    printf("                  ranks); without it each rank progresses independently.\n");
    printf("  --comm-only     Skip the GEMM, loop the collective.\n");
    printf("  --compute <m>   Compute phase: gemm|triad|eltwise (default gemm).\n");
    printf("  --cpasses <n>   Bandwidth-kernel launches per lap (duration knob).\n");
    printf("  --cbytes <b>    Bytes per working array (default 1 GiB).\n");
    printf("  -o 0|1          Overlap: 0 = sequential (collective completes before the GEMM),\n");
    printf("                  1 = overlapped (collective enqueued before the GEMM). Both\n");
    printf("                  complete each lap before the next one is queued.\n");
#endif
}

// ---------------------------------------------------------------------------
// Argument parser
// ---------------------------------------------------------------------------
static GemmArgs parseArguments(int argc, char **argv)
{
    GemmArgs args;
    args.device_id      = 0;
    args.matrix_size    = DEFAULT_SIZE;
    args.repeats        = DEFAULT_NTIMES;
    args.init_mode      = INIT_RANDOM;

    int co;
    opterr = 0;

    // m:c:p:o: consumed by parseCommArguments / parseSleepArguments where applicable
    while ((co = getopt(argc, argv, "hd:s:n:i:m:c:p:o:")) != -1) {
        switch (co) {

        case 'h':
            printHelp(argv[0]);
            exit(EXIT_SUCCESS);

        case 'd': {
            char *end; errno = 0;
            long val = strtol(optarg, &end, 10);
            if (*end != '\0' || errno != 0 || val < 0) {
                fprintf(stderr, "Invalid device ID for -d: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            args.device_id = (int)val;
            break;
        }

        case 's': {
            char *end; errno = 0;
            long val = strtol(optarg, &end, 10);
            if (*end != '\0' || errno != 0 || val <= 0) {
                fprintf(stderr, "Invalid matrix size for -s: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            args.matrix_size = (size_t)val;
            break;
        }

        case 'n': {
            char *end; errno = 0;
            long val = strtol(optarg, &end, 10);
            if (*end != '\0' || errno != 0 || val <= 0) {
                fprintf(stderr, "Invalid iteration count for -n: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            args.repeats = (int)val;
            break;
        }

        case 'i':
            if (strcmp(optarg, "random") == 0) {
                args.init_mode = INIT_RANDOM;
            } else if (strcmp(optarg, "constant") == 0) {
                args.init_mode = INIT_CONSTANT;
            } else {
                fprintf(stderr, "Invalid init mode for -i: '%s'. Use 'random' or 'constant'.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;

        case 'm': case 'c': case 'p': case 'o':
            break;  // comm/sleep-only flags — parsed in parseCommArguments / parseSleepArguments

        case '?':
            if (isprint(optopt))
                fprintf(stderr, "Unknown option `-%c'. Use -h for help.\n", optopt);
            else
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            exit(EXIT_FAILURE);

        default:
            abort();
        }
    }

    return args;
}

// ===========================================================================
// Communication arguments (gemm_comms_nccl)
// ===========================================================================

#ifdef COMM_MSG_SIZE
#  define DEFAULT_COMM_MSG_SIZE  ((size_t)(COMM_MSG_SIZE))
#else
#  define DEFAULT_COMM_MSG_SIZE  ((size_t)(32 * 1024 * 1024))  // 32 MB
#endif

typedef enum {
    COMM_ALLREDUCE = 0,
    COMM_ALLGATHER,
    COMM_CE_ALLGATHER,   // NCCL 2.28+ copy-engine (zero-CTA) all-gather (needs ncclMemAlloc + window reg)
    COMM_REDUCESCATTER   // ncclReduceScatter (SM-based): input nranks*msg, own-chunk output
} CommPattern;

typedef struct {
    size_t      msg_size;       // bytes per collective
    CommPattern pattern;        // allreduce or allgather
    int         overlap;        // 1 = overlap GEMM with collective (default), 0 = sequential
} CommArgs;

// Parse a size argument with optional K/M/G suffix (case-insensitive).
// "-m 32M" → 33554432, "-m 1G" → 1073741824, "-m 4096" → 4096.
static size_t parse_size_with_suffix(const char *s, const char *flag)
{
    char *end; errno = 0;
    long long val = strtoll(s, &end, 10);
    if (errno != 0 || val < 0 || end == s) {
        fprintf(stderr, "Invalid size for %s: %s\n", flag, s);
        exit(EXIT_FAILURE);
    }
    if (*end == 'k' || *end == 'K') { val *= 1024LL;                end++; }
    else if (*end == 'm' || *end == 'M') { val *= 1024LL * 1024;    end++; }
    else if (*end == 'g' || *end == 'G') { val *= 1024LL * 1024 * 1024; end++; }
    if (*end != '\0') {
        fprintf(stderr, "Invalid size suffix for %s: '%s' (use K, M, or G)\n", flag, s);
        exit(EXIT_FAILURE);
    }
    return (size_t)val;
}

static CommArgs parseCommArguments(int argc, char **argv)
{
    CommArgs args;
    args.msg_size         = DEFAULT_COMM_MSG_SIZE;
    args.pattern          = COMM_ALLREDUCE;
    args.overlap          = 1;

    optind = 1;
    opterr = 0;

    int co;
    while ((co = getopt(argc, argv, "m:p:o:")) != -1) {
        switch (co) {
        case 'm':
            args.msg_size = parse_size_with_suffix(optarg, "-m");
            break;
        case 'p':
            if (strcmp(optarg, "allgather") == 0)          args.pattern = COMM_ALLGATHER;
            else if (strcmp(optarg, "allreduce") == 0)     args.pattern = COMM_ALLREDUCE;
            else if (strcmp(optarg, "ce_allgather") == 0)   args.pattern = COMM_CE_ALLGATHER;
            else if (strcmp(optarg, "reducescatter") == 0)    args.pattern = COMM_REDUCESCATTER;
            else {
                fprintf(stderr, "Invalid pattern for -p: '%s'. Use allreduce|allgather|reducescatter|ce_allgather.\n", optarg);
                exit(EXIT_FAILURE);
            }
            break;
        case 'o':
            args.overlap = atoi(optarg);  // 0=seq, 1=unbounded bg, 2=bounded (comm-completion barrier)
            break;
        case '?':
        default:
            break;
        }
    }
    return args;
}

// ===========================================================================
// Benchmark-specific long options (gemm_comms_nccl): --barrier, --comm-only,
// --compute, --cpasses, --cbytes. Parsed and stripped from argv BEFORE the
// getopt-based parsers above run (getopt would abort on unknown options).
// ===========================================================================

typedef enum { COMPUTE_GEMM = 0, COMPUTE_TRIAD, COMPUTE_ELTWISE } ComputeMode;

typedef struct {
  int         use_barrier;      // --barrier: per-lap rank sync (1-elem NCCL AllReduce)
  int         comm_only;        // --comm-only: skip the GEMM, loop the collective
  ComputeMode compute_mode;     // --compute gemm|triad|eltwise
  int         compute_passes;   // --cpasses: bandwidth-kernel launches per lap
  size_t      compute_bytes;    // --cbytes: bytes per working array
} BenchOpts;

static BenchOpts parse_bench_opts(int &argc, char **argv)
{
    BenchOpts o;
    o.use_barrier    = 0;
    o.comm_only      = 0;
    o.compute_mode   = COMPUTE_GEMM;
    o.compute_passes = 1;
    o.compute_bytes  = 1024ull * 1024ull * 1024ull;

    auto take_val = [&](int &i) -> const char * {
        const char *v = (i + 1 < argc) ? argv[i + 1] : nullptr;
        if (!v) { fprintf(stderr, "missing value for %s\n", argv[i]); exit(EXIT_FAILURE); }
        for (int j = i; j < argc - 2; j++) argv[j] = argv[j + 2];
        argc -= 2;
        return v;
    };

    for (int i = 1; i < argc; ) {
        if (strcmp(argv[i], "--barrier") == 0) {
            o.use_barrier = 1;
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--;
        } else if (strcmp(argv[i], "--comm-only") == 0) {
            o.comm_only = 1;
            for (int j = i; j < argc - 1; j++) argv[j] = argv[j + 1];
            argc--;
        } else if (strcmp(argv[i], "--compute") == 0) {
            const char *cm = take_val(i);
            if      (strcmp(cm, "triad") == 0)   o.compute_mode = COMPUTE_TRIAD;
            else if (strcmp(cm, "eltwise") == 0) o.compute_mode = COMPUTE_ELTWISE;
            else if (strcmp(cm, "gemm") == 0)    o.compute_mode = COMPUTE_GEMM;
            else { fprintf(stderr, "--compute must be gemm|triad|eltwise\n"); exit(EXIT_FAILURE); }
        } else if (strcmp(argv[i], "--cpasses") == 0) {
            o.compute_passes = atoi(take_val(i));
        } else if (strcmp(argv[i], "--cbytes") == 0) {
            o.compute_bytes = strtoull(take_val(i), nullptr, 10);
        } else {
            i++;
        }
    }
    return o;
}

// ---------------------------------------------------------------------------
// Sleep arguments (gemm_sleep)
// ---------------------------------------------------------------------------

typedef struct {
    double sleep_sec;   // seconds per lap (-m); 0 = disable (GEMM only)
} SleepArgs;

static SleepArgs parseSleepArguments(int argc, char **argv)
{
    SleepArgs args;
    args.sleep_sec = 1.0;

#ifdef SLEEP_DEFAULT_SEC
    args.sleep_sec = (double)SLEEP_DEFAULT_SEC;
#endif

    optind = 1;
    opterr = 0;

    int co;
    while ((co = getopt(argc, argv, "m:p:b:k:")) != -1) {
        switch (co) {
        case 'm': {
            char *end; errno = 0;
            double val = strtod(optarg, &end);
            if (*end != '\0' || errno != 0 || val < 0.0) {
                fprintf(stderr, "Invalid sleep seconds for -m: %s\n", optarg);
                exit(EXIT_FAILURE);
            }
            args.sleep_sec = val;
            break;
        }
        case 'p':
        case 'b':
            break;
        case '?':
        default:
            break;
        }
    }
    return args;
}

#endif // CLI_H

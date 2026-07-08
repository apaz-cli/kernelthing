/* libktgpu.so -- how kernelthing puts each process on its own GPU.
 *
 * kernelthing runs many processes that want a GPU at once: agents building and
 * testing kernels, plus the benchmark that scores them. Two processes sharing
 * a card corrupts timings and can OOM it, so device use must be exclusive,
 * one process tree per card at a time.
 *
 * Rather than trusting each process to cooperate, the orchestrator injects
 * this library into everything it spawns via LD_PRELOAD (inherited by all
 * descendants), along with one variable describing the cards it may use:
 *
 *     KERNELTHING_GPU_POOL=UUID=/lock/path;UUID2=/lock/path2;...
 *
 * Each entry is a physical GPU (its nvidia-smi UUID is stable, unlike CUDA
 * indices, which depend on each process's device masking) paired with a
 * lockfile that stands for exclusive use of that card. The lockfiles are
 * named and created by kernelthing/gpupool.py, and sandboxed processes get
 * them bind-mounted at the same path, so every process on the box that
 * targets a card -- across sandboxes and kernelthing instances -- serializes
 * on one inode. This shim is the only code that flocks them.
 *
 * That variable is the whole interface: the pool comes in through
 * KERNELTHING_GPU_POOL, and the chosen card goes out through
 * CUDA_VISIBLE_DEVICES. Life of a shimmed process:
 *
 *   load time        The constructor kt_init() inspects CUDA_VISIBLE_DEVICES.
 *                    If it names one of the pool's cards, an ancestor claimed
 *                    that card and this process inherits it (see "children").
 *                    Any other value is discarded -- blanked, so CUDA sees no
 *                    devices at all. This is the fail-closed default: a
 *                    process launched with CUDA_VISIBLE_DEVICES=0 to dodge
 *                    the lock gets nothing.
 *
 *   first CUDA use   The hooks at the bottom of this file interpose the two
 *                    roads into the GPU driver: dlopen() of libcuda (how the
 *                    CUDA runtime, ctypes-style bindings, and torch load it)
 *                    and cuInit() (for programs linked against libcuda
 *                    directly). The first one hit runs ktgpu_acquire():
 *                    flock() a free card's lockfile and set
 *                    CUDA_VISIBLE_DEVICES=<its UUID>, so the driver
 *                    initializes seeing exactly that one card. If every card
 *                    is busy, block until one frees.
 *
 *   lifetime         The lock fd is left open for the life of the process.
 *                    The kernel drops a flock when its holder exits or dies,
 *                    so a crashed or SIGKILLed process can never wedge a card.
 *
 *   children         The claim is recorded in CUDA_VISIBLE_DEVICES itself,
 *                    which children inherit; a child seeing a pool UUID there
 *                    at load time uses that card WITHOUT locking again. This
 *                    matters because e.g. torch touches CUDA in a parent
 *                    process and runs the real work in a spawned child -- on
 *                    a one-card pool the child would otherwise deadlock
 *                    waiting on its own parent's lock. (Inheriting is
 *                    cooperative, not a security boundary: kernelthing's
 *                    command guard separately blocks agents from setting
 *                    CUDA_VISIBLE_DEVICES or any KERNELTHING_* variable.)
 *
 *   CPU-only work    A process that never reaches for the driver never
 *                    triggers any of this and never holds a card -- including
 *                    torch imports and builds, whose libraries load libcuda
 *                    only to resolve symbols (see the dlopen hook's nesting
 *                    rule). Hooking the road to the driver itself -- rather
 *                    than guessing from command lines -- is what makes the
 *                    exclusion exactly as wide as real device use.
 *
 * If KERNELTHING_GPU_POOL is unset, the shim is inert: it touches nothing and
 * the process behaves as if the library were never loaded.
 *
 * Building this needs no CUDA headers or libraries: the hooks chain to the
 * real implementations through dlsym(RTLD_NEXT) with minimal function-pointer
 * typedefs, and where no real implementation exists (a machine without CUDA)
 * they are harmless stubs.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#define KT_POOL_ENV "KERNELTHING_GPU_POOL"
#define KT_POOL_MAX 64 /* max cards parsed from the pool */

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_acquired = 0; /* this process has a card (own lock or inherited) */
static int g_lock_fd = -1; /* open forever: the flock lives as long as the fd */

/* Parse "UUID=/lock/path;..." into parallel arrays, in place (the uuid/path
 * pointers point into buf, which must outlive them). Malformed entries are
 * skipped. Returns the number of cards found. */
static int kt_parse_pool(char *buf, char *uuid[], char *path[]) {
  int n = 0;
  char *save = NULL;
  for (char *tok = strtok_r(buf, ";", &save); tok && n < KT_POOL_MAX;
       tok = strtok_r(NULL, ";", &save)) {
    char *eq = strchr(tok, '=');
    if (!eq || eq == tok || !eq[1]) continue;
    *eq = '\0';
    uuid[n] = tok;
    path[n] = eq + 1;
    n++;
  }
  return n;
}

/* Try to take the exclusive flock on one lockfile. Returns the open fd holding
 * the lock, or -1. When block==0 a busy lock fails immediately (LOCK_NB);
 * when block!=0 we wait for it. */
static int kt_try_lock(const char *path, int block) {
  int fd = open(path, O_RDWR | O_CREAT, 0666);
  if (fd < 0) return -1;
  if (flock(fd, LOCK_EX | (block ? 0 : LOCK_NB)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

/* Make *uuid* this process's card: pin CUDA_VISIBLE_DEVICES so the real CUDA
 * library sees only that device -- and so descendants inherit the claim.
 *
 * setenv() updates the C environment only. A Python process that claims
 * mid-run keeps its stale os.environ snapshot (CVD="" from kt_init), with two
 * consequences: torch's NVML-based device_count() reports 0 there (though
 * is_available() and allocation work), and passing env=dict(os.environ) to a
 * subprocess launders the claim away -- the child locks a *second* card, which
 * on a one-card pool means deadlocking on its own parent. Children spawned
 * with the default (C) environment inherit correctly. */
static void kt_claim(const char *uuid, int lock_fd) {
  g_lock_fd = lock_fd;
  setenv("CUDA_VISIBLE_DEVICES", uuid, 1);
  g_acquired = 1;
}

/* Lock a card from the pool for this process. Called once, from the first
 * hook hit below (the inherited-card case was already settled at load time
 * by kt_init, which sets g_acquired). Returns 0 once a card is pinned, -1 if
 * there is no pool (shim inert) or nothing could be locked. Exposed
 * (non-static) so the test harness can drive it without a real CUDA install. */
int ktgpu_acquire(void) {
  if (g_acquired) return 0;

  const char *pool = getenv(KT_POOL_ENV);
  if (!pool || !*pool) return -1;

  char *buf = strdup(pool);
  if (!buf) return -1;
  char *uuid[KT_POOL_MAX], *path[KT_POOL_MAX];
  int n = kt_parse_pool(buf, uuid, path);

  /* First pass: probe every card without waiting; take the first free one. */
  for (int i = 0; i < n; i++) {
    int fd = kt_try_lock(path[i], 0);
    if (fd >= 0) {
      kt_claim(uuid[i], fd);
      free(buf);
      return 0;
    }
  }

  /* Every card busy: wait on the first one -- everyone queues on a
   * deterministic card rather than racing across all of them. */
  if (n > 0) {
    int fd = kt_try_lock(path[0], 1);
    if (fd >= 0) {
      kt_claim(uuid[0], fd);
      free(buf);
      return 0;
    }
  }

  free(buf);
  return -1;
}

static void kt_acquire_once(void) { (void)ktgpu_acquire(); }
static void kt_ensure(void) { pthread_once(&g_once, kt_acquire_once); }

/* Runs when the library is loaded -- before main(), before any CUDA call.
 * Decides what the inherited CUDA_VISIBLE_DEVICES means:
 *
 *   - names a card in the pool -> an ancestor claimed it; adopt it. We take
 *     no lock and hold no fd (the ancestor's open fd keeps the flock alive).
 *   - anything else -> discard it (fail-closed: no device is reachable until
 *     ktgpu_acquire() pins the one card we lock).
 *
 * Only the environment is settled here -- locking itself waits for the first
 * sign of CUDA use, so CPU-only processes never take a card. */
__attribute__((constructor)) static void kt_init(void) {
  const char *pool = getenv(KT_POOL_ENV);
  if (!pool || !*pool) return; /* no pool configured -> shim is inert */

  const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
  if (cvd && *cvd) {
    char *buf = strdup(pool);
    if (buf) {
      char *uuid[KT_POOL_MAX], *path[KT_POOL_MAX];
      int n = kt_parse_pool(buf, uuid, path);
      for (int i = 0; i < n; i++) {
        if (strcmp(cvd, uuid[i]) == 0) {
          g_acquired = 1; /* inherited card, already pinned */
          free(buf);
          return;
        }
      }
      free(buf);
    }
  }
  setenv("CUDA_VISIBLE_DEVICES", "", 1);
}

/* ---- The interposed entry points --------------------------------------------
 *
 * Acquisition must happen before the driver initializes: libcuda reads
 * CUDA_VISIBLE_DEVICES once, in the process's first cuInit(), and the device
 * set is fixed from then on. Two hooks cover every road there:
 *
 * dlopen -- everything not linked against libcuda loads it with
 * dlopen("libcuda.so.1"): the CUDA runtime (the shared libcudart AND the
 * static cudart that nvcc links by default), ctypes/numba-style bindings,
 * torch's driver-API shim. Interposing CUDA symbols cannot catch these -- they
 * look driver entry points up with dlsym(handle, ...), which never consults
 * the preload -- but they all share this one step, loading the driver library,
 * so we acquire just before a load of libcuda.
 *
 * With one exception: a dlopen of libcuda from inside another dlopen (depth
 * tracked per thread below) is a library constructor resolving driver symbols
 * for later, not use. libcublasLt and friends do exactly this when torch
 * preloads them at import, and claiming there would put a card under every
 * process that merely imports torch -- including the nvcc build workers that
 * are kept off the GPU lock on purpose. Skipping nested loads is safe because
 * every path that goes on to *use* the driver re-enters dlopen("libcuda.so.1")
 * at top level first (dlopen of an already-loaded library just bumps its
 * refcount): the runtime re-dlopens it when lazy-initializing, and
 * ctypes-style users open it directly. Acquiring can block on a busy pool;
 * that happens before chaining, so we never sleep holding the loader's locks.
 *
 * cuInit -- a program linked against libcuda directly never dlopens it (ld.so
 * maps it at startup). Its driver-API calls resolve through the dynamic
 * linker, where the preload wins, and every driver call requires a prior
 * cuInit(), so hooking cuInit alone covers this road. */

#define KT_NEXT(fnptr_type, name)                    \
  static fnptr_type real = NULL;                     \
  if (!real) real = (fnptr_type)dlsym(RTLD_NEXT, name)

static __thread int g_dlopen_depth = 0;

typedef void *(*dlopen_t)(const char *, int);
void *dlopen(const char *filename, int flags) {
  KT_NEXT(dlopen_t, "dlopen");
  if (filename && g_dlopen_depth == 0) {
    const char *base = strrchr(filename, '/');
    base = base ? base + 1 : filename;
    if (strncmp(base, "libcuda.so", 10) == 0) kt_ensure();
  }
  if (!real) return NULL;
  g_dlopen_depth++;
  void *handle = real(filename, flags);
  g_dlopen_depth--;
  return handle;
}

typedef int (*cuInit_t)(unsigned int);
int cuInit(unsigned int flags) {
  kt_ensure();
  KT_NEXT(cuInit_t, "cuInit");
  return real ? real(flags) : 0;
}

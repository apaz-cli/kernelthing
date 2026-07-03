/* kernelthing GPU-lock shim (LD_PRELOAD).
 *
 * Replaces the old cooperative `gpu_run.sh` wrapper. Instead of trusting the
 * agent to prefix GPU commands, this library is injected into every process the
 * agent spawns (LD_PRELOAD is set by the orchestrator, inherited through bash).
 * It interposes the CUDA entry points that begin device use and, on the FIRST
 * such call in a process, atomically claims a free GPU from the pool:
 *
 *   1. read KERNELTHING_GPU_POOL  ("UUID=/lock/path;UUID2=/lock/path2;...")
 *   2. flock(LOCK_EX|LOCK_NB) each lockfile in turn; take the first that is free
 *      (block on the first candidate if every card is busy)
 *   3. keep the lock fd open for the process lifetime -> the flock is held until
 *      the process exits (or dies), then released by the kernel
 *   4. setenv CUDA_VISIBLE_DEVICES=<uuid> so the real CUDA init only sees that
 *      one card, then chain to the real symbol
 *
 * Purely CPU processes never call any CUDA entry point, so they never take a
 * lock -- this is why the shim is precise where bash-command parsing was not.
 *
 * The lockfiles are the same per-UUID files kernelthing/gpulock.py creates, so a
 * shimmed agent process and the Python scorer serialize against each other on
 * the identical inode.
 *
 * No CUDA headers required: the real symbols are resolved with dlsym(RTLD_NEXT)
 * through minimal function-pointer typedefs.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

#define KT_POOL_ENV "KERNELTHING_GPU_POOL"
#define KT_HELD_ENV "KERNELTHING_GPU_HELD"
#define KT_MAX_UUID 128

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_lock_fd = -1;             /* held open for process lifetime (-1 if inherited) */
static int g_acquired = 0;             /* resolved a card: own flock OR inherited from ancestor */
static char g_uuid[KT_MAX_UUID] = {0}; /* chosen device UUID, for diagnostics */

/* Try to claim one lockfile. Returns an open, flock-held fd or -1.
 * When block!=0 the flock is blocking (waits for the card); otherwise LOCK_NB. */
static int kt_try_lock(const char *path, int block) {
  int fd = open(path, O_RDWR | O_CREAT, 0666);
  if (fd < 0) return -1;
  int op = LOCK_EX | (block ? 0 : LOCK_NB);
  if (flock(fd, op) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

/* Core allocation: parse the pool, claim a card, pin CUDA_VISIBLE_DEVICES.
 * Idempotent-safe to call once (guarded by pthread_once in the hooks). Exposed
 * (non-static) so the test harness can drive it without a real CUDA install. */
int ktgpu_acquire(void) {
  if (g_acquired) return 0; /* already resolved a card for this process */

  /* Re-entrancy across a process subtree: if an ancestor already flocked a card
   * (it exports the UUID in KERNELTHING_GPU_HELD, inherited through exec), adopt
   * that same card instead of taking a *second* flock. This is essential because
   * pygpubench/torch run the real GPU work in a spawned child: the parent that
   * first touches CUDA (e.g. torch's device-capability probe when building)
   * holds the lock for the whole subtree, so on a single-card pool the child
   * would otherwise deadlock waiting on a card its own parent is holding. We
   * don't hold an fd here -- the ancestor's fd keeps the flock alive. */
  const char *held = getenv(KT_HELD_ENV);
  if (held && *held) {
    snprintf(g_uuid, sizeof(g_uuid), "%s", held);
    setenv("CUDA_VISIBLE_DEVICES", held, 1);
    g_acquired = 1;
    return 0;
  }

  const char *pool = getenv(KT_POOL_ENV);
  if (!pool || !*pool) return -1; /* no pool configured -> fail open (no-op) */

  char *buf = strdup(pool);
  if (!buf) return -1;

  char *first_uuid = NULL, *first_path = NULL;

  /* Non-blocking pass: take the first free card. */
  char *save = NULL;
  for (char *tok = strtok_r(buf, ";", &save); tok; tok = strtok_r(NULL, ";", &save)) {
    char *eq = strchr(tok, '=');
    if (!eq) continue;
    *eq = '\0';
    char *uuid = tok, *path = eq + 1;
    if (!*uuid || !*path) continue;
    if (!first_uuid) {
      first_uuid = uuid;
      first_path = path;
    }
    int fd = kt_try_lock(path, 0);
    if (fd >= 0) {
      g_lock_fd = fd;
      g_acquired = 1;
      snprintf(g_uuid, sizeof(g_uuid), "%s", uuid);
      setenv("CUDA_VISIBLE_DEVICES", uuid, 1);
      setenv(KT_HELD_ENV, uuid, 1); /* descendants adopt this card, don't re-lock */
      free(buf);
      return 0;
    }
  }

  /* Every card busy: block on the first candidate (matches gpulock.pick_free_gpu). */
  if (first_path) {
    int fd = kt_try_lock(first_path, 1);
    if (fd >= 0) {
      g_lock_fd = fd;
      g_acquired = 1;
      snprintf(g_uuid, sizeof(g_uuid), "%s", first_uuid);
      setenv("CUDA_VISIBLE_DEVICES", first_uuid, 1);
      setenv(KT_HELD_ENV, first_uuid, 1); /* descendants adopt this card, don't re-lock */
      free(buf);
      return 0;
    }
  }

  free(buf);
  return -1;
}

static void kt_acquire_once(void) { (void)ktgpu_acquire(); }
static void kt_ensure(void) { pthread_once(&g_once, kt_acquire_once); }

/* Runs at library load, before main() and before any CUDA call.
 *
 * If an ancestor already holds a card (KERNELTHING_GPU_HELD inherited through
 * exec), pin CUDA_VISIBLE_DEVICES to it right away so this process sees that one
 * card even before its first hooked call fires.
 *
 * Otherwise, if a pool is configured, neutralize any CUDA_VISIBLE_DEVICES
 * inherited from the launching command (e.g. an agent trying
 * `CUDA_VISIBLE_DEVICES=0 python ...` to dodge the lock): blank it so no device
 * is reachable until ktgpu_acquire() pins the one card we lock. When no pool is
 * configured the shim is inert and leaves the environment untouched. */
__attribute__((constructor)) static void kt_init(void) {
  const char *held = getenv(KT_HELD_ENV);
  if (held && *held) {
    setenv("CUDA_VISIBLE_DEVICES", held, 1);
    return;
  }
  const char *pool = getenv(KT_POOL_ENV);
  if (pool && *pool) setenv("CUDA_VISIBLE_DEVICES", "", 1);
}

/* --- interposed CUDA entry points --------------------------------------------
 * Each hook claims a card (once) before chaining to the real implementation.
 * We hook both the driver API (cuInit and the cuGetProcAddress trampoline used
 * by modern libcudart) and the earliest runtime API calls torch makes. */

#define KT_NEXT(fnptr_type, name)                    \
  static fnptr_type real = NULL;                     \
  if (!real) real = (fnptr_type)dlsym(RTLD_NEXT, name)

typedef int (*cuInit_t)(unsigned int);
int cuInit(unsigned int flags) {
  kt_ensure();
  KT_NEXT(cuInit_t, "cuInit");
  return real ? real(flags) : 0;
}

typedef int (*cudaGetDeviceCount_t)(int *);
int cudaGetDeviceCount(int *count) {
  kt_ensure();
  KT_NEXT(cudaGetDeviceCount_t, "cudaGetDeviceCount");
  return real ? real(count) : 0;
}

typedef int (*cudaSetDevice_t)(int);
int cudaSetDevice(int device) {
  kt_ensure();
  KT_NEXT(cudaSetDevice_t, "cudaSetDevice");
  return real ? real(device) : 0;
}

typedef int (*cudaMalloc_t)(void **, size_t);
int cudaMalloc(void **ptr, size_t size) {
  kt_ensure();
  KT_NEXT(cudaMalloc_t, "cudaMalloc");
  return real ? real(ptr, size) : 0;
}

/* libcudart (CUDA 11.3+) fetches driver symbols through cuGetProcAddress rather
 * than the dynamic linker, which would bypass our cuInit hook; intercept the
 * lookup and hand back our wrapper instead. Both the v1 and v2 ABIs are covered. */
typedef int (*cuGetProcAddress_t)(const char *, void **, int, unsigned long long);
int cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion,
                     unsigned long long flags) {
  KT_NEXT(cuGetProcAddress_t, "cuGetProcAddress");
  int rc = real ? real(symbol, pfn, cudaVersion, flags) : -1;
  if (symbol && pfn && strcmp(symbol, "cuInit") == 0) *pfn = (void *)cuInit;
  return rc;
}

typedef int (*cuGetProcAddress_v2_t)(const char *, void **, int, unsigned long long,
                                     void *);
int cuGetProcAddress_v2(const char *symbol, void **pfn, int cudaVersion,
                        unsigned long long flags, void *status) {
  KT_NEXT(cuGetProcAddress_v2_t, "cuGetProcAddress_v2");
  int rc = real ? real(symbol, pfn, cudaVersion, flags, status) : -1;
  if (symbol && pfn && strcmp(symbol, "cuInit") == 0) *pfn = (void *)cuInit;
  return rc;
}

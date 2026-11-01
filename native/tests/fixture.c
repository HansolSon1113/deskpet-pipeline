/* Contract fixture only. No model execution or hardware is involved. */
#include "deskpet_native.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
typedef struct { double ready; int cancelled; } job;
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec+t.tv_nsec/1e9; }
static void *create(const char *config, char **error) {
    if (strstr(config, "missing-backend")) { *error=strdup("BACKEND_UNSUPPORTED"); return NULL; }
    return malloc(1);
}
static void *submit(void *model, const char *input, char **error) {
    (void)model; (void)input; (void)error;
    job *j=calloc(1,sizeof(job)); j->ready=now()+0.04; return j;
}
static int poll(void *handle, const char **result, const char **error) {
    job *j=handle; if(now()<j->ready) return 0;
    if(j->cancelled) { *error="CANCELLED"; return -1; }
    *result="{\"text\":\"native fixture\"}"; return 1;
}
static void cancel(void *handle) { ((job *)handle)->cancelled=1; }
static void free_error(char *error) { free(error); }
static const dp_backend_api api={1,create,free,submit,poll,cancel,free,free_error};
const dp_backend_api *deskpet_backend_v1(void) { return &api; }

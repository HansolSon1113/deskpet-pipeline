#ifndef DESKPET_NATIVE_H
#define DESKPET_NATIVE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Backend-neutral ABI. JSON contains model inputs/results, never runtime objects.
   poll: 0=pending, 1=completed, -1=failed. Strings live until job release.
   release_job drops the caller reference; submitted work retains model/buffers. */
typedef struct dp_backend_api {
    uint32_t version;
    void *(*create)(const char *configuration, char **error);
    void (*destroy)(void *model);
    void *(*submit)(void *model, const char *input, char **error);
    int (*poll)(void *job, const char **result, const char **error);
    void (*cancel)(void *job);
    void (*release_job)(void *job);
    void (*free_error)(char *error);
} dp_backend_api;
typedef const dp_backend_api *(*dp_backend_entry)(void);
typedef struct dp_library dp_library;
dp_library *dp_library_open(const char *path, char **error);
void dp_library_close(dp_library *library);
void dp_string_free(char *value);
void *dp_model_create(dp_library *, const char *, char **error);
void dp_model_destroy(dp_library *, void *);
void *dp_model_submit(dp_library *, void *, const char *, char **error);
int dp_model_poll(dp_library *, void *, const char **result, const char **error);
void dp_model_cancel(dp_library *, void *);
void dp_model_release_job(dp_library *, void *);
#ifdef __cplusplus
}
#endif
#endif

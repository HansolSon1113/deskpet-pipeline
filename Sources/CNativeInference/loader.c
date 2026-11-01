#include "deskpet_native.h"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
struct dp_library { void *handle; const dp_backend_api *api; };
dp_library *dp_library_open(const char *path, char **error) {
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) { *error = strdup(dlerror()); return NULL; }
    dp_backend_entry entry = (dp_backend_entry)dlsym(handle, "deskpet_backend_v1");
    const dp_backend_api *api = entry ? entry() : NULL;
    if (!api || api->version != 1 || !api->create || !api->destroy || !api->submit ||
        !api->poll || !api->cancel || !api->release_job || !api->free_error) {
        *error = strdup("BACKEND_ABI_MISMATCH"); dlclose(handle); return NULL;
    }
    dp_library *lib = calloc(1, sizeof(*lib)); lib->handle = handle; lib->api = api; return lib;
}
void dp_library_close(dp_library *l) { if (l) { dlclose(l->handle); free(l); } }
void dp_string_free(char *v) { free(v); }
static void copy_error(dp_library *l, char **out, char *native) {
    if (native) { *out = strdup(native); l->api->free_error(native); }
}
void *dp_model_create(dp_library *l, const char *c, char **e) {
    char *native = NULL; void *m = l->api->create(c, &native); copy_error(l,e,native); return m;
}
void dp_model_destroy(dp_library *l, void *m) { l->api->destroy(m); }
void *dp_model_submit(dp_library *l, void *m, const char *i, char **e) {
    char *native = NULL; void *j = l->api->submit(m,i,&native); copy_error(l,e,native); return j;
}
int dp_model_poll(dp_library *l, void *j, const char **r, const char **e) { return l->api->poll(j,r,e); }
void dp_model_cancel(dp_library *l, void *j) { l->api->cancel(j); }
void dp_model_release_job(dp_library *l, void *j) { l->api->release_job(j); }

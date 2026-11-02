#include "deskpet_native.h"
#include <sys/resource.h>
#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <stdio.h>
#include <unistd.h>
#endif

dp_process_memory dp_read_process_memory(void) {
    dp_process_memory result = {0};
#ifdef __APPLE__
    task_vm_info_data_t info = {0};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count) == KERN_SUCCESS) {
        result.resident_bytes = info.resident_size;
        result.resident_available = 1;
        if (count >= TASK_VM_INFO_REV1_COUNT) {
            result.footprint_bytes = info.phys_footprint;
            result.footprint_available = 1;
        }
    }
#else
    FILE *file = fopen("/proc/self/statm", "r");
    unsigned long total_pages = 0, resident_pages = 0;
    long page_size = sysconf(_SC_PAGESIZE);
    if (file) {
        if (fscanf(file, "%lu %lu", &total_pages, &resident_pages) == 2 && page_size > 0) {
            result.resident_bytes = (uint64_t)resident_pages * (uint64_t)page_size;
            result.resident_available = 1;
        }
        fclose(file);
    }
#endif
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
#ifdef __APPLE__
        result.peak_resident_bytes = (uint64_t)usage.ru_maxrss;
#else
        result.peak_resident_bytes = (uint64_t)usage.ru_maxrss * 1024;
#endif
        result.peak_available = 1;
    }
    return result;
}

#define _GNU_SOURCE  // 必须放在文件最开头以启用 GNU 扩展
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>
#include <string.h>
#include <openssl/sha.h>
#include <math.h>
#include <sched.h>
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <x86intrin.h>
#include <unistd.h>
#include <sched.h>

// 校准TSC频率（单位：Hz）
static double calibrate_tsc_freq() {
    struct timespec start, end;
    uint64_t tsc_start, tsc_end;
    const double calibration_time = 0.1; // 校准时间（秒）

    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    tsc_start = __rdtsc();
    usleep(calibration_time * 1e6); // 睡眠固定时间
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    tsc_end = __rdtsc();

    double real_time = (end.tv_sec - start.tv_sec) + 
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    return (tsc_end - tsc_start) / real_time;
}

// 计算Tboot = Tw - (TSC_current / TSC_freq)
double calculate_tboot(double tsc_freq) {
    struct timespec tw;
    uint64_t tsc_current;
    double tboot;

    // 获取当前时间Tw（单调时钟）
    clock_gettime(CLOCK_MONOTONIC_RAW, &tw);
    double tw_seconds = tw.tv_sec + tw.tv_nsec / 1e9;

    // 获取当前TSC值
    tsc_current = __rdtsc();

    // 计算Tboot
    tboot = tw_seconds - (tsc_current / tsc_freq);
    return tboot;
}

int main() {
    // 绑定到CPU核心0（避免多核TSC不同步）
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    if (pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
        perror("Failed to set CPU affinity");
        return 1;
    }

    // 校准TSC频率
    double tsc_freq = calibrate_tsc_freq();
    if (tsc_freq <= 0) {
        fprintf(stderr, "TSC频率校准失败\n");
        return 1;
    }
    printf("校准后的TSC频率: %.2f GHz\n", tsc_freq / 1e9);

    // 计算Tboot
    double tboot = calculate_tboot(tsc_freq);
    printf("系统启动时间 Tboot: %.9f 秒\n", tboot);

    return 0;
}
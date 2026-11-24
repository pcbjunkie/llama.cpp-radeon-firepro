#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

struct local_vk_profile {
    const char * name;
    std::atomic<unsigned long long> calls{0}, ns{0};
    explicit local_vk_profile(const char * name) : name(name) {}
    ~local_vk_profile() {
        if (calls) std::fprintf(stderr, "VKFUNC %s %llu %llu\n",
                                name, calls.load(), ns.load());
    }
};
struct local_vk_scope {
    using clock = std::chrono::steady_clock;
    local_vk_profile * stats;
    clock::time_point start;
    explicit local_vk_scope(local_vk_profile * stats)
        : stats(stats), start(stats ? clock::now() : clock::time_point{}) {}
    ~local_vk_scope() {
        if (stats) {
            stats->ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(
                clock::now() - start).count(), std::memory_order_relaxed);
            stats->calls.fetch_add(1, std::memory_order_relaxed);
        }
    }
};
#define LOCAL_VK_PROFILE() \
    static const bool trace = std::getenv("LOCAL_VK_PROFILE") != nullptr; \
    static local_vk_profile stats(__func__); \
    local_vk_scope scope(trace ? &stats : nullptr)

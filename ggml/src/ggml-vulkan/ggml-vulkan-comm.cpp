#include "local-function-profile.h"
#include "ggml-vulkan-common.h"

#if !defined(_WIN32)
#include <fcntl.h>
#include <chrono>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <cerrno>
#if defined(__linux__)
#include <pthread.h>
#endif

// local: prompt-sized rounds (f16 size and up) go as q8_0, half the bytes through host memory, whose DMA
// bandwidth limits prompt processing on the R720 (LOCAL_VK_COMM_Q8=0 sends f16). Measured 2026-11-24, 4x S7150
// at 110 W, pp512: 365 -> 400 tok/s at depth 0, 178 -> 183 at 16K; perplexity 5.0423 -> 5.0451 (+-0.115).
static const bool ggml_vk_comm_q8_wire = [] { const char * e = getenv("LOCAL_VK_COMM_Q8"); return e == nullptr || atoi(e) != 0; }();
// local experiment: LOCAL_VK_COMM_F16_MIN overrides the size from which the wire format is f16
static const size_t ggml_vk_comm_f16_min_size = [] { const char * e = getenv("LOCAL_VK_COMM_F16_MIN"); return e ? (size_t) atoll(e) : (size_t) 1024 * 1024; }();

// follows ops ggml-backend-meta may delay a reduction across
// TODO: the required precision should come from the meta backend instead.
static bool ggml_vk_comm_needs_f32(const ggml_tensor * t, int depth = 0) {
    if (t == nullptr || depth > 64) {
        return false;
    }
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return ggml_prec_get_acc(t) == GGML_PREC_BF16 || ggml_prec_get_acc(t) == GGML_PREC_F32;
        case GGML_OP_ADD:
            return ggml_vk_comm_needs_f32(t->src[0], depth + 1) || ggml_vk_comm_needs_f32(t->src[1], depth + 1);
        case GGML_OP_MUL:
        case GGML_OP_ADD_ID:
            return ggml_vk_comm_needs_f32(t->src[0], depth + 1);
        case GGML_OP_VIEW:
            return ggml_vk_comm_needs_f32(t->src[0], depth + 1);
        default:
            return false;
    }
}

// host_bridge: device i signals done[i], a host thread signals go[i] once all peers of i are done
// sync_fd: a host thread exchanges SYNC_FD semaphores between peers, the wait resolves in the kernel
enum class vk_comm_sync {
    host_bridge,
    sync_fd,
    gpu_event, // local: flag in host memory, host thread gates the submit
    timeline,  // local: one shared timeline semaphore per device, imported into every peer
    gpu_spin,  // local: flag in host memory, a shader on each peer spins on it, no host involvement
};

struct vk_comm_device {
    ggml_backend_t                     backend  = nullptr;
    ggml_backend_vk_context *          ctx      = nullptr;
    void *                             host_ptr = nullptr;
    size_t                             host_size = 0; // local: mmap length of host_ptr
    std::vector<ggml_backend_buffer_t> host_buf; // host_ptr imported into each peer

    struct {
        PFN_vkGetSemaphoreFdKHR    get_fd    = nullptr;
        PFN_vkImportSemaphoreFdKHR import_fd = nullptr;
        vk::Semaphore              sem;
        std::vector<vk::Semaphore> imp;      // [peer], pooled, reused only after ggml_vk_graph_cleanup
    } sfd;

    struct {
        vk::Semaphore done;
        vk::Semaphore go;
    } bridge;

    vk::Event ev[2];

    ggml_backend_buffer_t      scratch = nullptr; // local: device buffer swept to evict L2 in the spin wait
    ggml_backend_buffer_t      stage   = nullptr; // local: device buffer for the f16 partial before its DMA to host
    ggml_backend_buffer_t      peer_f32 = nullptr; // local: device buffer, the peers' q8_0 slots dequantized (n - 1 slots of cap)
    // local: LOCAL_VK_COMM_TS=1 times the steps of each round on this device's timeline
    vk::QueryPool              ts_pool;
    bool                       ts_used[64] = {};
    uint64_t                   ts_prev_end = 0;   // timestamp at the end of the previous harvested round
    uint64_t                   ts_prev_round = 0;
    double                     ts_sum[5] = {};    // us: compute before A, data out, flag, wait, adds
    uint64_t                   ts_cnt = 0;
    vk::Semaphore              tl;      // own timeline, signalled with the round number
    std::vector<vk::Semaphore> tl_imp;  // [peer] the peer's timeline imported here
};

struct ggml_backend_vk_comm_context {
    std::vector<vk_comm_device> devs;
    vk_comm_sync                mode     = vk_comm_sync::host_bridge;
    size_t                      align    = 4096;
    size_t                      cap      = 0;
    uint64_t                    round    = 0;
    ggml_context *              tctx     = nullptr;
    ggml_tensor *               slot     = nullptr;
    std::vector<ggml_tensor *>  peer_slots; // local: one view per peer for the fused multi_add
    bool                        disabled = false;

    std::thread                 thread;
    std::chrono::nanoseconds    spin{ 200'000 };
    std::atomic<uint64_t>       submitted{ 0 };
    std::atomic<uint64_t>       imported{ 0 };
    std::atomic<bool>           sleeping{ false };
    std::atomic<bool>           stop{ false };
    std::mutex                  mutex;
    std::condition_variable     cv;

    std::atomic<uint64_t>       rec_round[8] = {}; // local: last round recorded per device (diagnostics)
    std::atomic<uint32_t>       arrived[2] = { 0, 0 }; // local: per-device path, A submits per round parity
    bool                        threaded = false;      // local: set while each device has its own host thread
    std::thread                 tracer;                // local: LOCAL_VK_SPIN_TRACE sampler
    std::atomic<bool>           tracer_stop{ false };
};

// local: host buffer layout per device: [ring of slots][ring of flag pages][debug]
static constexpr size_t LOCAL_RING       = 16;
// GCN3 cache line: readers poll one word per line, so a 128-byte line only doubled the bytes the flag fill DMAs to
// host memory each round (2026-11-24, decode tg128 48.2 -> 49.1 tok/s with 160 lines)
static constexpr size_t LOCAL_FLAG_LINE  = 64;
static constexpr size_t LOCAL_FLAG_LINES = 256;
static constexpr size_t LOCAL_FLAG_PAGE  = LOCAL_FLAG_LINE * LOCAL_FLAG_LINES;
// lines of a flag page written and polled per round (LOCAL_VK_SPIN_FILL_LINES overrides). Measured 2026-11-24 in
// decode (tg128, clocks pinned, parallel polling), 128-byte lines: 256 47.4, 192 48.0, 160 48.1, 128 47.7-48.6
// tok/s, 64 collapses (readers keep hitting stale L2 lines and sweep); 64-byte lines: 160 48.8-49.1, 128 49.1,
// 96 collapses. 160 keeps a margin above that edge.
static constexpr size_t LOCAL_FLAG_USE_LINES = 160;

// data slots are reused after LOCAL_SLOT_RING rounds: a device can be at most about one round ahead of the
// slowest reader, so a short ring is enough. It is kept short because the buffers are imported into every
// GPU as userptr memory, and a large mapping (16 x 4 MB prompt-sized slots) made later decode stall.
// The flag pages keep the longer ring: polling relies on lines not having been read recently.
static constexpr size_t LOCAL_SLOT_RING  = 4;
static size_t ggml_vk_comm_slot_off(size_t cap, uint64_t r) { return (r % LOCAL_SLOT_RING) * cap; }
static size_t ggml_vk_comm_flag_off(size_t cap, uint64_t r) { return LOCAL_SLOT_RING * cap + (r % LOCAL_RING) * LOCAL_FLAG_PAGE; }
static size_t ggml_vk_comm_dbg_off(size_t cap)              { return LOCAL_SLOT_RING * cap + LOCAL_RING * LOCAL_FLAG_PAGE; }

static void ggml_vk_comm_free_buffers(ggml_backend_vk_comm_context * comm) {
    for (vk_comm_device & d : comm->devs) {
        if (d.stage) {
            ggml_backend_buffer_free(d.stage);
            d.stage = nullptr;
        }
        if (d.peer_f32) {
            ggml_backend_buffer_free(d.peer_f32);
            d.peer_f32 = nullptr;
        }
        for (ggml_backend_buffer_t & buf : d.host_buf) {
            ggml_backend_buffer_free(buf);
            buf = nullptr;
        }
        if (d.host_ptr) {
            munmap(d.host_ptr, d.host_size);
        }
        d.host_ptr  = nullptr;
        d.host_size = 0;
    }
    comm->cap = 0;
}

static bool ggml_backend_vk_comm_ensure(ggml_backend_vk_comm_context * comm, size_t nbytes) {
    if (comm->disabled) {
        return false;
    }
    // grow when needed; shrink back when the rounds are much smaller (decode after a prompt): every submit
    // on amdgpu revalidates all userptr pages of the device (~4 us per MB on this box), used or not, so a
    // prompt-sized ring kept around halved decode speed. Driver allocations shared as dma-bufs would avoid
    // that, but an imported dma-buf gets implicit sync, so each submit waits for the exporter's running
    // job, which deadlocks with the spin wait.
    if (nbytes <= comm->cap && (comm->cap <= (64u << 10) || nbytes * 16 > comm->cap)) {
        return true;
    }
    auto fail = [comm]() {
        GGML_LOG_WARN("ggml_vulkan: shared host buffer allocation failed, using generic all-reduce\n");
        comm->disabled = true;
        return false;
    };
    const auto local_t0 = std::chrono::steady_clock::now();
    for (vk_comm_device & d : comm->devs) {
        ggml_backend_synchronize(d.backend);
    }
    const auto local_t1 = std::chrono::steady_clock::now();
    ggml_vk_comm_free_buffers(comm);
    struct local_ensure_log {
        std::chrono::steady_clock::time_point t0, t1; size_t from, to;
        ~local_ensure_log() {
            if (std::getenv("LOCAL_VK_COMM_LOG")) {
                const auto t2 = std::chrono::steady_clock::now();
                fprintf(stderr, "COMMREALLOC cap %zu -> %zu: sync %.1f ms, realloc %.1f ms\n", from, to,
                    std::chrono::duration<double, std::milli>(t1 - t0).count(), std::chrono::duration<double, std::milli>(t2 - t1).count());
            }
        }
    } local_el { local_t0, local_t1, comm->cap, nbytes };

    const size_t n   = comm->devs.size();
    const size_t cap = GGML_PAD(nbytes, comm->align);
    for (size_t i = 0; i < n; i++) {
        vk_comm_device & d = comm->devs[i];
        const size_t total = GGML_PAD(ggml_vk_comm_dbg_off(cap) + 4096, comm->align);
        // own mapping, out of reach of NUMA balancing and khugepaged: their unmaps make the amdgpu
        // userptr notifier wait for every job using the pages, including peers spinning on our flag
        void * hp = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (hp == MAP_FAILED) {
            d.host_ptr = nullptr;
            return fail();
        }
        d.host_ptr  = hp;
        d.host_size = total;
        madvise(hp, total, MADV_NOHUGEPAGE);
        // MPOL_PREFERRED with an empty node mask: local allocation, and a VMA policy without MPOL_F_MOF.
        // local experiment: LOCAL_VK_COMM_NUMA=1 prefers the NUMA node of the device that writes the slot
        static const int local_comm_numa = [] { const char * e = getenv("LOCAL_VK_COMM_NUMA"); return e ? atoi(e) : 0; }();
        int node = -1;
        if (local_comm_numa) {
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(ggml_backend_get_device(d.backend), &props);
            if (props.device_id) {
                const std::string path = std::string("/sys/bus/pci/devices/") + props.device_id + "/numa_node";
                if (FILE * f = fopen(path.c_str(), "r")) {
                    if (fscanf(f, "%d", &node) != 1) {
                        node = -1;
                    }
                    fclose(f);
                }
            }
        }
        unsigned long nodemask = node >= 0 ? (1ul << node) : 0;
        if (syscall(SYS_mbind, hp, total, 1 /* MPOL_PREFERRED */, node >= 0 ? &nodemask : nullptr, node >= 0 ? 64 : 0, 0) != 0) {
            GGML_LOG_WARN("ggml_vulkan: mbind of comm buffer failed (%d), NUMA balancing may stall spin waits\n", errno);
        }
        if (local_comm_numa && getenv("LOCAL_VK_COMM_LOG")) {
            fprintf(stderr, "COMMNUMA dev%zu node %d\n", i, node);
        }
        if (mlock(hp, total) != 0) {
            // not fatal: only avoids reclaim
        }
        memset((char *) d.host_ptr + LOCAL_SLOT_RING * cap, 0, total - LOCAL_SLOT_RING * cap);
        for (size_t j = 0; j < n; j++) {
            d.host_buf[j] = ggml_backend_vk_device_buffer_from_host_ptr(
                ggml_backend_get_device(comm->devs[j].backend), d.host_ptr, total, total);
            if (!d.host_buf[j]) {
                return fail();
            }
        }
        // local: prompt-sized slots (f16 wire) are converted in VRAM and copied to host by DMA
        if (cap >= ggml_vk_comm_f16_min_size) {
            d.stage = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(d.backend), cap);
            if (ggml_vk_comm_q8_wire) {
                d.peer_f32 = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(d.backend), cap * (n - 1));
            }
        }
    }
    comm->cap = cap;
    return true;
}

static void ggml_vk_comm_set_slot(ggml_tensor * slot, const ggml_tensor * src, ggml_type type, ggml_backend_buffer_t buf, size_t offs) {
    slot->type  = type;
    slot->nb[0] = ggml_type_size(type);
    for (int d = 0; d < GGML_MAX_DIMS; d++) {
        slot->ne[d] = src->ne[d];
    }
    // block types (q8_0 wire): a row is ne0 / block size blocks
    slot->nb[1] = ggml_row_size(type, slot->ne[0]);
    for (int d = 2; d < GGML_MAX_DIMS; d++) {
        slot->nb[d] = slot->nb[d - 1] * slot->ne[d - 1];
    }
    slot->buffer = buf;
    slot->data   = (char *) ggml_backend_buffer_get_base(buf) + offs;
}

static const bool local_comm_ts = getenv("LOCAL_VK_COMM_TS") != nullptr;
#define LOCAL_TS_RING 64
#define LOCAL_TS_N 5
// timestamps per round: 0 A start (compute done), 1 data in host slot, 2 flag written, 3 wait done, 4 adds done
static void ggml_vk_comm_ts_harvest(vk_comm_device & d, uint32_t slot, uint64_t round) {
    if (!d.ts_used[slot]) {
        return;
    }
    uint64_t data[LOCAL_TS_N * 2];
    const vk::Result r = d.ctx->device->device.getQueryPoolResults(d.ts_pool, slot * LOCAL_TS_N, LOCAL_TS_N, sizeof(data), data, 16,
        vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWithAvailability);
    d.ts_used[slot] = false;
    if (r != vk::Result::eSuccess) {
        return;
    }
    for (int k = 0; k < LOCAL_TS_N; k++) {
        if (data[2 * k + 1] == 0) {
            return;
        }
    }
    const double us = d.ctx->device->properties.limits.timestampPeriod / 1000.0;
    if (d.ts_prev_round + 1 == round && d.ts_prev_end && data[0] > d.ts_prev_end) {
        d.ts_sum[0] += (data[0] - d.ts_prev_end) * us;
        for (int k = 1; k < LOCAL_TS_N; k++) {
            d.ts_sum[k] += (data[2 * k] - data[2 * (k - 1)]) * us;
        }
        d.ts_cnt++;
    }
    d.ts_prev_end = data[2 * (LOCAL_TS_N - 1)];
    d.ts_prev_round = round;
}
static void ggml_vk_comm_ts(vk_comm_device & d, vk_context & c, uint64_t round, uint32_t idx) {
    if (!local_comm_ts) {
        return;
    }
    if (!d.ts_pool) {
        d.ts_pool = d.ctx->device->device.createQueryPool({ {}, vk::QueryType::eTimestamp, LOCAL_TS_RING * LOCAL_TS_N });
        d.ctx->device->device.resetQueryPool(d.ts_pool, 0, LOCAL_TS_RING * LOCAL_TS_N);
    }
    const uint32_t slot = round % LOCAL_TS_RING;
    if (idx == 0) {
        // the slot was last written LOCAL_TS_RING rounds ago: read it before reuse (rounds are harvested in order)
        ggml_vk_comm_ts_harvest(d, slot, round >= LOCAL_TS_RING ? round - LOCAL_TS_RING : 0);
        c->s->buffer->buf.resetQueryPool(d.ts_pool, slot * LOCAL_TS_N, LOCAL_TS_N);
        d.ts_used[slot] = true;
    }
    c->s->buffer->buf.writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, d.ts_pool, slot * LOCAL_TS_N + idx);
}

static void ggml_vk_comm_submit(ggml_backend_vk_context * ctx, vk_context & c, vk::Semaphore signal, uint64_t value) {
    ggml_vk_ctx_end(c);
    for (auto & cpy : c->in_memcpys) {
        memcpy(cpy.dst, cpy.src, cpy.n);
    }
    for (auto & mset : c->memsets) {
        memset(mset.dst, mset.val, mset.n);
    }
    c->in_memcpys.clear();
    c->memsets.clear();
    if (signal) {
        c->seqs.back().back().signal_semaphores.push_back({ signal, value });
    }
    ggml_vk_submit(c, {});
    ctx->submit_pending = true;
    ctx->compute_ctx.reset();
}

static void ggml_vk_comm_notify_submitted(ggml_backend_vk_comm_context * comm, uint64_t round) {
    comm->submitted.store(round);
    if (comm->sleeping.load()) {
        std::lock_guard<std::mutex> lock(comm->mutex);
        comm->cv.notify_one();
    }
}

// false on stop
static bool ggml_vk_comm_wait_submitted(ggml_backend_vk_comm_context * comm, uint64_t r) {
    using clock = std::chrono::steady_clock;
    for (const auto t0 = clock::now(); comm->submitted.load() < r;) {
        if (comm->stop.load()) {
            return false;
        }
        if (clock::now() - t0 < comm->spin) {
            YIELD();
            continue;
        }
        comm->sleeping.store(true);
        if (comm->submitted.load() < r) {
            std::unique_lock<std::mutex> lock(comm->mutex);
            comm->cv.wait(lock, [&] { return comm->submitted.load() >= r || comm->stop.load(); });
        }
        comm->sleeping.store(false);
    }
    return true;
}

// signals go[i] once all peers of i are done; false on stop
static bool ggml_vk_comm_bridge_release(ggml_backend_vk_comm_context * comm, uint64_t r, std::vector<uint64_t> & seen, std::vector<bool> & sent) {
    using clock = std::chrono::steady_clock;
    const size_t n = comm->devs.size();
    std::fill(sent.begin(), sent.end(), false);
    size_t left = n;
    for (const auto t0 = clock::now();;) {
        size_t pending = n;
        for (size_t i = 0; i < n; i++) {
            if (sent[i]) {
                continue;
            }
            bool ready = true;
            for (size_t j = 0; j < n && ready; j++) {
                if (j != i && seen[j] < r) {
                    const vk_comm_device & p = comm->devs[j];
                    seen[j] = p.ctx->device->device.getSemaphoreCounterValue(p.bridge.done);
                    ready   = seen[j] >= r;
                    pending = ready ? pending : j;
                }
            }
            if (ready) {
                const vk_comm_device & d = comm->devs[i];
                d.ctx->device->device.signalSemaphore({ d.bridge.go, r });
                sent[i] = true;
                left--;
            }
        }
        if (left == 0) {
            return true;
        }
        if (clock::now() - t0 < comm->spin) {
            YIELD();
            continue;
        }
        const vk_comm_device &      p = comm->devs[pending];
        const vk::SemaphoreWaitInfo wi{ {}, 1, &p.bridge.done, &r };
        while (p.ctx->device->device.waitSemaphores(wi, 100'000'000) != vk::Result::eSuccess) {
            if (comm->stop.load()) {
                return false;
            }
        }
    }
}

static void ggml_vk_comm_bridge_thread(ggml_backend_vk_comm_context * comm) {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "ggml-vk-comm");
#endif
    std::vector<uint64_t> seen(comm->devs.size(), 0);
    std::vector<bool>     sent(comm->devs.size());
    try {
        for (uint64_t r = 1;; r++) {
            if (!ggml_vk_comm_wait_submitted(comm, r) || !ggml_vk_comm_bridge_release(comm, r, seen, sent)) {
                return;
            }
        }
    } catch (const vk::SystemError & e) {
        GGML_ABORT("ggml_vulkan: all-reduce bridge failed: %s", e.what());
    }
}

static void ggml_vk_comm_bridge_init(ggml_backend_vk_comm_context * comm) {
    vk::SemaphoreTypeCreateInfo stci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo     sci{};
    sci.pNext = &stci;
    for (vk_comm_device & d : comm->devs) {
        d.bridge.done = d.ctx->device->device.createSemaphore(sci);
        d.bridge.go   = d.ctx->device->device.createSemaphore(sci);
    }
    comm->thread = std::thread(ggml_vk_comm_bridge_thread, comm);
}

static void ggml_vk_comm_bridge_free(vk_comm_device & d) {
    if (d.bridge.done) {
        d.ctx->device->device.destroySemaphore(d.bridge.done);
    }
    if (d.bridge.go) {
        d.ctx->device->device.destroySemaphore(d.bridge.go);
    }
}

static void ggml_vk_comm_barrier_to_host(vk_context & c) {
    const vk::MemoryBarrier to_host{ vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eHostRead };
    c->s->buffer->buf.pipelineBarrier(c->p->q->stage_flags, vk::PipelineStageFlagBits::eHost, {}, { to_host }, {}, {});
}

// the peers' slots were made available to the host before the wait resolved, the submit predates that
static void ggml_vk_comm_barrier_from_host(vk_context & c) {
    const vk::MemoryBarrier from_host{ vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead };
    c->s->buffer->buf.pipelineBarrier(vk::PipelineStageFlagBits::eHost, c->p->q->stage_flags, {}, { from_host }, {}, {});
}

static void ggml_vk_comm_bridge_signal(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
    const vk_comm_device & d = comm->devs[i];
    ggml_vk_comm_barrier_to_host(c);
    ggml_vk_comm_submit(d.ctx, c, d.bridge.done, round);
}

static void ggml_vk_comm_bridge_wait(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
    LOCAL_VK_PROFILE();
    c->s->wait_semaphores.push_back({ comm->devs[i].bridge.go, round });
    ggml_vk_comm_barrier_from_host(c);
}

static bool ggml_vk_comm_sfd_supported(vk_comm_device & d) {
    const vk_device & dev = d.ctx->device;
    const bool mesa = dev->driver_id == vk::DriverId::eMesaRadv ||
                      dev->driver_id == vk::DriverId::eIntelOpenSourceMESA ||
                      dev->driver_id == vk::DriverId::eMesaNvk;
    if (!mesa || !dev->external_semaphore) {
        return false;
    }
    d.sfd.get_fd    = (PFN_vkGetSemaphoreFdKHR)    dev->device.getProcAddr("vkGetSemaphoreFdKHR");
    d.sfd.import_fd = (PFN_vkImportSemaphoreFdKHR) dev->device.getProcAddr("vkImportSemaphoreFdKHR");
    if (!d.sfd.get_fd || !d.sfd.import_fd) {
        return false;
    }
    vk::PhysicalDeviceExternalSemaphoreInfo esi{ vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd };
    vk::ExternalSemaphoreProperties esp = dev->physical_device.getExternalSemaphoreProperties(esi);
    const vk::ExternalSemaphoreFeatureFlags need =
        vk::ExternalSemaphoreFeatureFlagBits::eExportable | vk::ExternalSemaphoreFeatureFlagBits::eImportable;
    return (esp.externalSemaphoreFeatures & need) == need &&
           (esp.compatibleHandleTypes & vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd);
}

// Mesa moves a queue to its submit thread on the first wait-before-signal, taking the CS ioctls off this thread
static void ggml_vk_comm_sfd_enable_submit_thread(const vk_device & dev) {
    vk::SemaphoreTypeCreateInfo stci{ vk::SemaphoreType::eTimeline, 0 };
    vk::SemaphoreCreateInfo     sci{};
    sci.pNext = &stci;
    const vk::Semaphore sem   = dev->device.createSemaphore(sci);
    const vk::Fence     fence = dev->device.createFence({});
    const uint64_t      one   = 1;
    vk::TimelineSemaphoreSubmitInfo tsi{ 1, &one, 0, nullptr };
    const vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands;
    vk::SubmitInfo si{ 1, &sem, &stage, 0, nullptr, 0, nullptr };
    si.setPNext(&tsi);
    dev->compute_queue->handle->submit(si, fence);
    dev->device.signalSemaphore({ sem, 1 });
    VK_CHECK(dev->device.waitForFences({ fence }, true, UINT64_MAX), "ggml_vk_comm_sfd_enable_submit_thread", dev);
    dev->device.destroyFence(fence);
    dev->device.destroySemaphore(sem);
}

// Requires a submitted signal on sem; the export resets it.
static int ggml_vk_comm_sfd_export(vk_comm_device & d) {
    int fd = -1;
    VkSemaphoreGetFdInfoKHR gi{};
    gi.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    gi.semaphore  = static_cast<VkSemaphore>(d.sfd.sem);
    gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    const VkResult result = d.sfd.get_fd((VkDevice) d.ctx->device->device, &gi, &fd);
    if (result != VK_SUCCESS) {
        GGML_ABORT("ggml_vulkan: vkGetSemaphoreFdKHR failed: %d", (int) result);
    }
    return fd;
}

// fd -1 imports an already signalled payload
static void ggml_vk_comm_sfd_import(vk_comm_device & d, vk::Semaphore sem, int fd) {
    VkImportSemaphoreFdInfoKHR isi{};
    isi.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
    isi.semaphore  = static_cast<VkSemaphore>(sem);
    isi.flags      = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
    isi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
    isi.fd         = fd;
    const VkResult result = d.sfd.import_fd((VkDevice) d.ctx->device->device, &isi);
    if (result != VK_SUCCESS) {
        GGML_ABORT("ggml_vulkan: vkImportSemaphoreFdKHR failed: %d", (int) result);
    }
}

static void ggml_vk_comm_sfd_exchange(ggml_backend_vk_comm_context * comm, uint64_t r) {
    const size_t n = comm->devs.size();
    for (size_t j = 0; j < n; j++) {
        const int    fd   = ggml_vk_comm_sfd_export(comm->devs[j]);
        const size_t last = j == n - 1 ? n - 2 : n - 1;
        for (size_t i = 0; i < n; i++) {
            if (i == j) {
                continue;
            }
            const int imp_fd = fd < 0 || i == last ? fd : fcntl(fd, F_DUPFD_CLOEXEC, 0);
            GGML_ASSERT(imp_fd >= 0 || fd < 0);
            ggml_vk_comm_sfd_import(comm->devs[i], comm->devs[i].sfd.imp[j], imp_fd);
        }
    }
    comm->imported.store(r, std::memory_order_release);
}

static void ggml_vk_comm_sfd_thread(ggml_backend_vk_comm_context * comm) {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "ggml-vk-comm");
#endif
    for (uint64_t r = 1; ggml_vk_comm_wait_submitted(comm, r); r++) {
        ggml_vk_comm_sfd_exchange(comm, r);
    }
}

static void ggml_vk_comm_sfd_init(ggml_backend_vk_comm_context * comm) {
    vk::ExportSemaphoreCreateInfo esci{ vk::ExternalSemaphoreHandleTypeFlagBits::eSyncFd };
    vk::SemaphoreCreateInfo       sci{};
    sci.pNext = &esci;
    for (vk_comm_device & d : comm->devs) {
        d.sfd.sem = d.ctx->device->device.createSemaphore(sci);
        d.sfd.imp.assign(comm->devs.size(), {});
        ggml_vk_comm_sfd_enable_submit_thread(d.ctx->device);
    }
    comm->thread = std::thread(ggml_vk_comm_sfd_thread, comm);
}

static volatile uint32_t * ggml_vk_comm_flag(ggml_backend_vk_comm_context * comm, size_t j, uint64_t r) {
    return (volatile uint32_t *) ((char *) comm->devs[j].host_ptr + ggml_vk_comm_flag_off(comm->cap, r));
}

static void ggml_vk_comm_event_thread(ggml_backend_vk_comm_context * comm) {
#if defined(__linux__)
    pthread_setname_np(pthread_self(), "ggml-vk-comm");
#endif
    const size_t n = comm->devs.size();
    static const bool local_ev_log = std::getenv("LOCAL_VK_EVENT_LOG") != nullptr;
    for (uint64_t r = 1;; r++) {
        // rescue: never leave a GPU waiting on an event forever
        for (auto t0 = std::chrono::steady_clock::now(); comm->submitted.load() < r;) {
            if (comm->stop.load()) {
                return;
            }
            if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
                GGML_LOG_ERROR("ggml_vulkan: event all-reduce: round %llu not recorded after 2 s, submitted=%llu flags:", (unsigned long long) r, (unsigned long long) comm->submitted.load());
                for (size_t k = 0; k < n; k++) {
                    GGML_LOG_ERROR(" %u", *ggml_vk_comm_flag(comm, k, r));
                }
                GGML_LOG_ERROR("\n");
                for (size_t k = 0; k < n; k++) {
                    for (vk::Event & e : comm->devs[k].ev) {
                        comm->devs[k].ctx->device->device.setEvent(e);
                    }
                }
                t0 = std::chrono::steady_clock::now() + std::chrono::hours(1);
            }
            std::this_thread::sleep_for(std::chrono::microseconds(20));
        }
        if (local_ev_log) {
            GGML_LOG_INFO("EVT round %llu recorded\n", (unsigned long long) r);
        }
        // A(r) of every device is recorded; wait for the GPUs to write the flags
        const auto t0 = std::chrono::steady_clock::now();
        bool timed_out = false;
        for (size_t j = 0; j < n && !timed_out; j++) {
            while ((uint32_t) (*ggml_vk_comm_flag(comm, j, r) - (uint32_t) r) > 0x80000000u) {
                if (comm->stop.load(std::memory_order_relaxed)) {
                    return;
                }
                if (std::chrono::steady_clock::now() - t0 > std::chrono::seconds(2)) {
                    GGML_LOG_ERROR("ggml_vulkan: event all-reduce stuck at round %llu:", (unsigned long long) r);
                    for (size_t k = 0; k < n; k++) {
                        GGML_LOG_ERROR(" dev%zu=%u", k, *ggml_vk_comm_flag(comm, k, r));
                    }
                    GGML_LOG_ERROR("\n");
                    timed_out = true;
                    break;
                }
                YIELD();
            }
        }
        static const bool local_event_wait = std::getenv("LOCAL_VK_EVENT_WAIT") != nullptr;
        if (local_event_wait) {
            for (size_t i = 0; i < n; i++) {
                comm->devs[i].ctx->device->device.setEvent(comm->devs[i].ev[r & 1]);
            }
        }
        if (local_ev_log) {
            GGML_LOG_INFO("EVT round %llu set\n", (unsigned long long) r);
        }
        comm->imported.store(r, std::memory_order_release);
    }
}

static void ggml_vk_comm_event_init(ggml_backend_vk_comm_context * comm) {
    for (vk_comm_device & d : comm->devs) {
        for (vk::Event & e : d.ev) {
            e = d.ctx->device->device.createEvent({});
        }
    }
    comm->thread = std::thread(ggml_vk_comm_event_thread, comm);
}

static bool ggml_vk_comm_timeline_init(ggml_backend_vk_comm_context * comm) {
    const size_t n = comm->devs.size();
    for (vk_comm_device & d : comm->devs) {
        const vk_device & dev = d.ctx->device;
        auto get_fd    = (PFN_vkGetSemaphoreFdKHR)    dev->device.getProcAddr("vkGetSemaphoreFdKHR");
        auto import_fd = (PFN_vkImportSemaphoreFdKHR) dev->device.getProcAddr("vkImportSemaphoreFdKHR");
        if (!dev->external_semaphore || !get_fd || !import_fd) {
            return false;
        }
        d.sfd.get_fd = get_fd;
        d.sfd.import_fd = import_fd;
        vk::SemaphoreTypeCreateInfo   stci{ vk::SemaphoreType::eTimeline, 0 };
        vk::ExportSemaphoreCreateInfo esci{ vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd };
        esci.pNext = &stci;
        vk::SemaphoreCreateInfo sci{};
        sci.pNext = &esci;
        d.tl = dev->device.createSemaphore(sci);
        d.tl_imp.assign(n, {});
    }
    for (size_t j = 0; j < n; j++) {
        vk_comm_device & src = comm->devs[j];
        for (size_t i = 0; i < n; i++) {
            if (i == j) {
                continue;
            }
            vk_comm_device & dst = comm->devs[i];
            int fd = -1;
            VkSemaphoreGetFdInfoKHR gi{};
            gi.sType      = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
            gi.semaphore  = static_cast<VkSemaphore>(src.tl);
            gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            if (src.sfd.get_fd((VkDevice) src.ctx->device->device, &gi, &fd) != VK_SUCCESS) {
                return false;
            }
            vk::SemaphoreTypeCreateInfo stci{ vk::SemaphoreType::eTimeline, 0 };
            vk::SemaphoreCreateInfo     sci{};
            sci.pNext = &stci;
            dst.tl_imp[j] = dst.ctx->device->device.createSemaphore(sci);
            VkImportSemaphoreFdInfoKHR isi{};
            isi.sType      = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR;
            isi.semaphore  = static_cast<VkSemaphore>(dst.tl_imp[j]);
            isi.flags      = 0;
            isi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            isi.fd         = fd;
            if (dst.sfd.import_fd((VkDevice) dst.ctx->device->device, &isi) != VK_SUCCESS) {
                close(fd);
                return false;
            }
        }
    }
    // waits may be submitted before the matching signal
    for (vk_comm_device & d : comm->devs) {
        ggml_vk_comm_sfd_enable_submit_thread(d.ctx->device);
    }
    return true;
}

static void ggml_vk_comm_sfd_free(vk_comm_device & d) {
    if (d.sfd.sem) {
        d.ctx->device->device.destroySemaphore(d.sfd.sem);
    }
}

static void ggml_vk_comm_sfd_signal(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c) {
    vk_comm_device & d = comm->devs[i];
    ggml_vk_comm_barrier_to_host(c);
    ggml_vk_comm_submit(d.ctx, c, d.sfd.sem, 0);
    for (size_t j = 0; j < comm->devs.size(); j++) {
        d.sfd.imp[j] = j == i ? vk::Semaphore{} : ggml_vk_create_binary_semaphore(d.ctx);
    }
}

static void ggml_vk_comm_sfd_wait(ggml_backend_vk_comm_context * comm, size_t i, vk_context & c, uint64_t round) {
    LOCAL_VK_PROFILE();
    for (size_t j = 0; j < comm->devs.size(); j++) {
        if (j != i) {
            c->s->wait_semaphores.push_back({ comm->devs[i].sfd.imp[j], 0 });
        }
    }
    c->s->ready       = &comm->imported;
    c->s->ready_value = round;
    ggml_vk_comm_barrier_from_host(c);
}

void ggml_backend_vk_comm_free(void * comm_ctx) {
    ggml_backend_vk_comm_context * comm = static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
    if (comm->tracer.joinable()) {
        comm->tracer_stop.store(true);
        comm->tracer.join();
    }
    if (comm->mode == vk_comm_sync::gpu_spin && comm->cap && getenv("LOCAL_VK_SPIN_DEBUG")) {
        for (size_t i = 0; i < comm->devs.size(); i++) {
            const uint32_t * w = (const uint32_t *) ((char *) comm->devs[i].host_ptr + ggml_vk_comm_dbg_off(comm->cap));
            const uint32_t flag = *(const uint32_t *) ((char *) comm->devs[i].host_ptr + ggml_vk_comm_flag_off(comm->cap, comm->round));
            fprintf(stderr, "SPINDBG dev%zu own_flag=%u round=%u polls=%u sweeps=%u total_sweeps=%u waits=%u total_polls=%u comm_round=%llu avg_wait_us=%.1f\n", i, flag, w[16], w[0], w[1], w[2], w[3], w[4], (unsigned long long) comm->round, w[3] ? w[5] / 27.0 / w[3] : 0.0);
        }
    }
    // pending reductions need the thread
    for (vk_comm_device & d : comm->devs) {
        ggml_backend_synchronize(d.backend);
    }
    if (local_comm_ts) {
        for (size_t i = 0; i < comm->devs.size(); i++) {
            vk_comm_device & d = comm->devs[i];
            if (d.ts_cnt) {
                const double c = (double) d.ts_cnt;
                fprintf(stderr, "COMMTS dev%zu rounds %llu (us/round): compute %.1f | data out %.1f | flag %.1f | wait %.1f | adds %.1f\n", i,
                        (unsigned long long) d.ts_cnt, d.ts_sum[0] / c, d.ts_sum[1] / c, d.ts_sum[2] / c, d.ts_sum[3] / c, d.ts_sum[4] / c);
            }
            if (d.ts_pool) {
                d.ctx->device->device.destroyQueryPool(d.ts_pool);
            }
        }
    }
    if (comm->thread.joinable()) {
        {
            std::lock_guard<std::mutex> lock(comm->mutex);
            comm->stop.store(true);
        }
        comm->cv.notify_all();
        comm->thread.join();
    }
    for (vk_comm_device & d : comm->devs) {
        ggml_vk_comm_sfd_free(d);
        ggml_vk_comm_bridge_free(d);
        for (vk::Event & e : d.ev) {
            if (e) {
                d.ctx->device->device.destroyEvent(e);
            }
        }
        for (vk::Semaphore & sem : d.tl_imp) {
            if (sem) {
                d.ctx->device->device.destroySemaphore(sem);
            }
        }
        if (d.tl) {
            d.ctx->device->device.destroySemaphore(d.tl);
        }
        if (d.scratch) {
            ggml_backend_buffer_free(d.scratch);
        }
    }
    ggml_vk_comm_free_buffers(comm);
    if (comm->tctx) {
        ggml_free(comm->tctx);
    }
    delete comm;
}

void * ggml_backend_vk_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (vk_instance.allreduce_disabled || n_backends < 2) {
        return nullptr;
    }
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_vk(backends[i])) {
            return nullptr;
        }
    }
    ggml_backend_vk_comm_context * comm = new ggml_backend_vk_comm_context;
    comm->devs.resize(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        vk_comm_device & d = comm->devs[i];
        d.backend = backends[i];
        d.ctx     = (ggml_backend_vk_context *) backends[i]->context;
        d.host_buf.assign(n_backends, nullptr);
    }

    for (size_t i = 0; i < n_backends; i++) {
        const vk_device & dev = comm->devs[i].ctx->device;
        comm->align = std::max(comm->align, (size_t) dev->min_imported_host_pointer_alignment);
        if (!dev->external_memory_host) {
            GGML_LOG_WARN("ggml_vulkan: device %zu: VK_EXT_external_memory_host missing, using generic all-reduce\n", i);
            ggml_backend_vk_comm_free(comm);
            return nullptr;
        }
        if (dev->serialize_submissions) {
            GGML_LOG_WARN("ggml_vulkan: device %zu: GGML_VK_SERIALIZE_SUBMISSIONS set, using generic all-reduce\n", i);
            ggml_backend_vk_comm_free(comm);
            return nullptr;
        }
    }

    if (getenv("LOCAL_VK_COMM_BRIDGE") == nullptr && std::all_of(comm->devs.begin(), comm->devs.end(), ggml_vk_comm_sfd_supported)) {
        comm->mode = vk_comm_sync::sync_fd;
    }
    if (const char * e = getenv("LOCAL_VK_COMM_EVENTS"); e != nullptr && atoi(e) != 0) {
        comm->mode = vk_comm_sync::gpu_event;
    }
    // GCN3 L2 serves stale host memory to shaders, so the spin mode is opt-in and broken there
    // default on; LOCAL_VK_COMM_SPIN=0 falls back to the sync_fd path
    const bool want_spin = [] { const char * e = getenv("LOCAL_VK_COMM_SPIN"); return e == nullptr || atoi(e) != 0; }();
    bool want_timeline = false;
    if (const char * e = getenv("LOCAL_VK_COMM_TIMELINE"); e != nullptr && atoi(e) != 0) {
        want_timeline = comm->mode != vk_comm_sync::gpu_event;
    }

    if (const char * spin = getenv("GGML_VK_ALLREDUCE_SPIN_US")) {
        comm->spin = std::chrono::microseconds(std::max(0, atoi(spin)));
    }
    const ggml_init_params ip = { (1 + n_backends) * ggml_tensor_overhead(), nullptr, true };
    comm->tctx = ggml_init(ip);
    comm->slot = ggml_new_tensor_1d(comm->tctx, GGML_TYPE_F32, 1);
    for (size_t j = 0; j < n_backends; j++) {
        comm->peer_slots.push_back(ggml_new_tensor_1d(comm->tctx, GGML_TYPE_F32, 1));
    }

    try {
        if (want_spin && comm->mode != vk_comm_sync::gpu_event && n_backends <= 9 &&
            std::all_of(comm->devs.begin(), comm->devs.end(), [](const vk_comm_device & d) { return (bool) d.ctx->device->pipeline_comm_wait; })) {
            comm->mode = vk_comm_sync::gpu_spin;
            want_timeline = false;
            for (vk_comm_device & d : comm->devs) {
                d.scratch = ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(d.backend), 4u << 20);
                GGML_ASSERT(d.scratch);
            }
            if (const char * tf = getenv("LOCAL_VK_SPIN_TRACE")) {
                std::string path = tf;
                comm->tracer = std::thread([comm, path] {
                    // events per device, on the CPU clock: H = host thread recorded the round, F = the device's flag
                    // for that round is in host memory, W = its spin wait for the round finished
                    FILE * f = fopen(path.c_str(), "w");
                    const size_t n = comm->devs.size();
                    std::vector<uint32_t> last_w(n, 0), next_f(n, 1);
                    std::vector<uint64_t> last_h(n, 0);
                    size_t cap_seen = 0;
                    const auto t0 = std::chrono::steady_clock::now();
                    while (!comm->tracer_stop.load()) {
                        const size_t cap = comm->cap;
                        if (cap && comm->devs[0].host_ptr) {
                            if (cap != cap_seen) {
                                // ring realloc: resync to the current round
                                cap_seen = cap;
                                for (size_t i = 0; i < n; i++) {
                                    next_f[i] = (uint32_t) comm->rec_round[i].load() + 1;
                                }
                            }
                            const long long us = (long long) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count();
                            for (size_t i = 0; i < n; i++) {
                                const char * hp = (const char *) comm->devs[i].host_ptr;
                                const uint64_t h = comm->rec_round[i].load(std::memory_order_relaxed);
                                if (h != last_h[i]) {
                                    fprintf(f, "%lld %zu H %llu\n", us, i, (unsigned long long) h);
                                    last_h[i] = h;
                                }
                                for (int guard = 0; guard < 8; guard++) {
                                    const uint32_t fv = *(volatile const uint32_t *) (hp + ggml_vk_comm_flag_off(cap, next_f[i]));
                                    if (int32_t(fv - next_f[i]) < 0) {
                                        break;
                                    }
                                    fprintf(f, "%lld %zu F %u\n", us, i, next_f[i]);
                                    next_f[i]++;
                                }
                                const uint32_t w = ((volatile const uint32_t *) (hp + ggml_vk_comm_dbg_off(cap)))[16];
                                if (w != last_w[i]) {
                                    fprintf(f, "%lld %zu W %u\n", us, i, w);
                                    last_w[i] = w;
                                }
                            }
                        }
                        std::this_thread::sleep_for(std::chrono::microseconds(5));
                    }
                    fclose(f);
                });
            }
        }
        if (want_timeline) {
            if (ggml_vk_comm_timeline_init(comm)) {
                comm->mode = vk_comm_sync::timeline;
                GGML_LOG_INFO("ggml_vulkan: all-reduce uses shared timeline semaphores\n");
            } else {
                GGML_LOG_WARN("ggml_vulkan: shared timeline semaphores unavailable, using sync fd\n");
            }
        }
        if (comm->mode == vk_comm_sync::timeline || comm->mode == vk_comm_sync::gpu_spin) {
            // no host thread
        } else if (comm->mode == vk_comm_sync::gpu_event) {
            ggml_vk_comm_event_init(comm);
        } else if (comm->mode == vk_comm_sync::sync_fd) {
            ggml_vk_comm_sfd_init(comm);
        } else {
            ggml_vk_comm_bridge_init(comm);
        }
    } catch (const vk::SystemError & e) {
        GGML_LOG_WARN("ggml_vulkan: semaphore initialization failed: %s, using generic all-reduce\n", e.what());
        ggml_backend_vk_comm_free(comm);
        return nullptr;
    }
    return comm;
}

// A: partial to own host slot (round & 1), signal; B: wait on peers' A, add their slots; all A precede any B
// slot reuse: A(k) follows own B(k-1), which waited on peers' A(k-1), submitted after their B(k-2), the last reader
bool ggml_backend_vk_comm_allreduce_dev(void * comm_ctx, size_t i, ggml_tensor * t, uint64_t round);

bool ggml_backend_vk_comm_allreduce_tensor(void * comm_ctx, ggml_tensor ** tensors) {
    LOCAL_VK_PROFILE();
    ggml_backend_vk_comm_context * comm = static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
    const size_t n = comm->devs.size();

    if (comm->mode == vk_comm_sync::gpu_event || comm->mode == vk_comm_sync::timeline || comm->mode == vk_comm_sync::gpu_spin) {
        const int64_t ne0 = ggml_nelements(tensors[0]);
        for (size_t i = 0; i < n; i++) {
            const ggml_tensor * t = tensors[i];
            if (t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne0 || !ggml_is_contiguous(t) || !ggml_backend_buffer_is_vk(t->buffer)) {
                return false;
            }
        }
        if (ne0 == 0) {
            return true;
        }
        for (size_t i = 0; i < n; i++) {
            ggml_vk_local_flush_pending_reduce(comm->devs[i].ctx);
        }
        if (!ggml_backend_vk_comm_ensure(comm, ggml_nbytes(tensors[0]))) {
            return false;
        }
        const uint64_t round = ++comm->round;
        comm->threaded = false;
        for (size_t i = 0; i < n; i++) {
            ggml_backend_vk_comm_allreduce_dev(comm_ctx, i, tensors[i], round);
        }
        return true;
    }

    for (size_t i = 0; i < n; i++) {
        ggml_vk_local_flush_pending_reduce(comm->devs[i].ctx);
    }
    const int64_t ne = ggml_nelements(tensors[0]);
    for (size_t i = 0; i < n; i++) {
        const ggml_tensor * t = tensors[i];
        if (t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne || !ggml_is_contiguous(t) || !ggml_backend_buffer_is_vk(t->buffer)) {
            return false;
        }
        const ggml_backend_vk_context * ctx = comm->devs[i].ctx;
        if (!ctx->compute_ctx.expired() && !ctx->compute_ctx.lock()->out_memcpys.empty()) {
            return false;
        }
    }
    if (ne == 0) {
        return true;
    }
    // local timing experiment, results are wrong when set:
    //   1 = skip the reduction entirely, 2 = keep copy + adds but skip cross-device signal/wait/submit
    static const int local_fake_allreduce = [] {
        const char * e = std::getenv("LOCAL_VK_FAKE_ALLREDUCE");
        return e ? std::atoi(e) : 0;
    }();
    if (local_fake_allreduce == 1) {
        return true;
    }
    const size_t nbytes = ggml_nbytes(tensors[0]);
    if (!ggml_backend_vk_comm_ensure(comm, nbytes)) {
        return false;
    }

    const bool      f32_wire  = nbytes < ggml_vk_comm_f16_min_size || ggml_vk_comm_needs_f32(tensors[0]);
    const ggml_type wire      = f32_wire ? GGML_TYPE_F32 : GGML_TYPE_F16;
    const uint64_t  round     = ++comm->round;
    const size_t    slot_offs = ggml_vk_comm_slot_off(comm->cap, round);
    const size_t    slot_size = ne * ggml_type_size(wire);

    for (size_t i = 0; i < n; i++) {
        const vk_comm_device &    d   = comm->devs[i];
        ggml_backend_vk_context * ctx = d.ctx;
        ggml_tensor *             t   = tensors[i];
        vk_buffer & tbuf = ((ggml_backend_vk_buffer_context *) t->buffer->context)->dev_buffer;
        vk_buffer & hbuf = ((ggml_backend_vk_buffer_context *) d.host_buf[i]->context)->dev_buffer;
        const size_t toffs = vk_tensor_offset(t) + t->view_offs;

        vk_context c = ggml_vk_get_compute_ctx(ctx);
        ggml_vk_sync_buffers(ctx, c);
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            c->s->buffer->buf.fillBuffer(hbuf->buffer, slot_offs, GGML_PAD(slot_size, 4), 0);
            c->s->buffer->buf.fillBuffer(tbuf->buffer, toffs, nbytes, 0);
        } else if (wire == GGML_TYPE_F32) {
            ggml_vk_buffer_copy_async(c, hbuf, slot_offs, tbuf, toffs, nbytes);
        } else {
            ggml_vk_comm_set_slot(comm->slot, t, wire, d.host_buf[i], slot_offs);
            ggml_vk_cpy(ctx, c, t, comm->slot);
        }
        if (local_fake_allreduce == 2) {
            continue;
        }
        if (comm->mode == vk_comm_sync::sync_fd) {
            ggml_vk_comm_sfd_signal(comm, i, c);
        } else {
            ggml_vk_comm_bridge_signal(comm, i, c, round);
        }
    }
    if (local_fake_allreduce != 2) {
        ggml_vk_comm_notify_submitted(comm, round);
    }

    for (size_t i = 0; i < n; i++) {
        ggml_backend_vk_context * ctx = comm->devs[i].ctx;
        vk_context c = ggml_vk_get_compute_ctx(ctx);
        ctx->do_add_rms_partials = false;
        if (local_fake_allreduce == 2) {
            // no peer wait
        } else if (comm->mode == vk_comm_sync::sync_fd) {
            ggml_vk_comm_sfd_wait(comm, i, c, round);
        } else {
            ggml_vk_comm_bridge_wait(comm, i, c, round);
        }
        ggml_vk_sync_buffers(ctx, c);
        // local: f32 wire -> sum own partial and all peer slots in one multi_add dispatch
        static const bool local_no_fused_reduce = std::getenv("LOCAL_VK_NO_FUSED_REDUCE") != nullptr;
        vk_pipeline fused = (!local_no_fused_reduce && wire == GGML_TYPE_F32 && ctx->device->multi_add && !ctx->device->disable_fusion &&
                             n >= 2 && n + 1 <= MAX_PARAMETER_COUNT) ? ctx->device->pipeline_multi_add[n - 2] : nullptr;
        static const bool local_no_fold = std::getenv("LOCAL_VK_NO_FOLD_RESIDUAL") != nullptr;
        if (fused && !local_no_fold && ctx->local_tail == tensors[i] && ctx->local_tail_uses == 1) {
            // defer: the next graph adds the residual in the same dispatch (or the sum is flushed as is)
            GGML_ASSERT(!ctx->local_pending_active);
            uint32_t k = 1;
            for (size_t j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                ggml_tensor & sl = ctx->local_pending_slots[k++];
                sl = *comm->peer_slots[j];
                ggml_vk_comm_set_slot(&sl, tensors[i], wire, comm->devs[j].host_buf[i], slot_offs);
            }
            ctx->local_pending_n = k;
            ctx->local_pending_own = tensors[i];
            ctx->local_pending_active = true;
            ctx->local_pending_hold = false;
            ctx->local_pending_fold_node = nullptr;
        } else if (fused) {
            const ggml_tensor * srcs[MAX_PARAMETER_COUNT];
            uint32_t k = 0;
            srcs[k++] = tensors[i];
            for (size_t j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                ggml_vk_comm_set_slot(comm->peer_slots[j], tensors[i], wire, comm->devs[j].host_buf[i], slot_offs);
                srcs[k++] = comm->peer_slots[j];
            }
            srcs[k] = tensors[i];
            ggml_vk_multi_add_tensors(ctx, c, fused, srcs, k);
            ggml_vk_sync_buffers(ctx, c);
        }
        for (size_t j = 0; j < n && !fused; j++) {
            if (j == i) {
                continue;
            }
            ggml_vk_comm_set_slot(comm->slot, tensors[i], wire, comm->devs[j].host_buf[i], slot_offs);
            ggml_vk_add(ctx, c, tensors[i], comm->slot, tensors[i]);
            ggml_vk_sync_buffers(ctx, c);
        }
        // small reductions go out with the next graph's first submit
        if (local_fake_allreduce != 2 && (wire != GGML_TYPE_F32 || vk_perf_logger_enabled)) {
            ggml_vk_comm_submit(ctx, c, {}, 0);
        }
    }
    return true;
}

// local: per-device all-reduce, so each device can be driven by its own host thread.
// Reserves capacity and round numbers up front. Returns the first round, 0 if not available.
uint64_t ggml_backend_vk_comm_begin(void * comm_ctx, size_t max_nbytes, size_t n_rounds) {
    ggml_backend_vk_comm_context * comm = static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
    if (comm->mode == vk_comm_sync::host_bridge || n_rounds == 0) {
        return 0;
    }
    for (vk_comm_device & d : comm->devs) {
        ggml_vk_local_flush_pending_reduce(d.ctx);
    }
    if (!ggml_backend_vk_comm_ensure(comm, max_nbytes)) {
        return 0;
    }
    const uint64_t base = comm->round + 1;
    comm->round += n_rounds;
    comm->threaded = true;
    return base;
}

bool ggml_backend_vk_comm_allreduce_dev(void * comm_ctx, size_t i, ggml_tensor * t, uint64_t round) {
    ggml_backend_vk_comm_context * comm = static_cast<ggml_backend_vk_comm_context *>(comm_ctx);
    // local, timing experiments only (results are garbage): bit0 no wait dispatch, bit1 no data_in barrier,
    // bit2 no copy of the partial, bit3 no flag fill
    static const int local_spin_exp = [] { const char * e = std::getenv("LOCAL_VK_SPIN_EXP"); return e ? atoi(e) : 0; }();
    // local timing experiment (results are garbage) for the f16 wire: 1 = no copy of the partial to host, 2 = no peer adds, 4 = convert in VRAM but no DMA to host
    static const int local_comm_exp = [] { const char * e = std::getenv("LOCAL_VK_COMM_EXP"); return e ? atoi(e) : 0; }();
    const size_t n = comm->devs.size();
    vk_comm_device & d = comm->devs[i];
    ggml_backend_vk_context * ctx = d.ctx;

    const int64_t ne = ggml_nelements(t);
    GGML_ASSERT(t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_backend_buffer_is_vk(t->buffer));
    if (ne == 0) {
        return true;
    }
    static const int local_fake_allreduce = [] {
        const char * e = std::getenv("LOCAL_VK_FAKE_ALLREDUCE");
        return e ? std::atoi(e) : 0;
    }();
    if (local_fake_allreduce == 1) {
        return true;
    }
    const size_t nbytes = ggml_nbytes(t);
    GGML_ASSERT(nbytes <= comm->cap);

    ggml_vk_local_flush_pending_reduce(ctx);
    if (!ctx->compute_ctx.expired() && !ctx->compute_ctx.lock()->out_memcpys.empty()) {
        ggml_backend_synchronize(d.backend);
    }
    const bool ev_mode = comm->mode == vk_comm_sync::gpu_event;
    const bool tl_mode = comm->mode == vk_comm_sync::timeline;
    const bool sp_mode = comm->mode == vk_comm_sync::gpu_spin;
    if (sp_mode) {
        // a wait on the GPU timed out (peer dead or stalled): the reduced data is wrong, stop here
        const uint32_t * w = (const uint32_t *) ((char *) d.host_ptr + ggml_vk_comm_dbg_off(comm->cap));
        if (__atomic_load_n(&w[8], __ATOMIC_ACQUIRE) != 0) {
            for (size_t j = 0; j < n; j++) {
                const uint32_t fv = *(volatile const uint32_t *) ((char *) comm->devs[j].host_ptr + ggml_vk_comm_flag_off(comm->cap, w[10]));
                const volatile uint32_t * wj = (const volatile uint32_t *) ((char *) comm->devs[j].host_ptr + ggml_vk_comm_dbg_off(comm->cap));
                fprintf(stderr, "SPINTO dev%zu: flag[wait_round]=%u recorded=%llu gpu_last_wait=%u timeouts=%u to_round=%u to_ticks=%u\n", j, fv, (unsigned long long) comm->rec_round[j].load(), wj[16], wj[8], wj[10], wj[9]);
            }
            GGML_ABORT("vulkan comm: spin wait on dev%zu timed out (LOCAL_VK_SPIN_LIMIT_US), a peer did not signal: ticks=%u wait_round=%u peer_k=%u polls=%u sweeps=%u now_round=%llu",
                       i, w[9], w[10], w[11], w[12], w[13], (unsigned long long) round);
        }
    }
    // own sfd.sem and imp[] of the previous round must be exchanged before reuse
    while (!ev_mode && !tl_mode && !sp_mode && comm->imported.load(std::memory_order_acquire) + 1 < round) {
        YIELD();
    }

    const bool      f32_wire  = nbytes < ggml_vk_comm_f16_min_size || ggml_vk_comm_needs_f32(t);
    const bool      q8_wire   = !f32_wire && ggml_vk_comm_q8_wire && d.peer_f32 && t->ne[0] % 32 == 0;
    const ggml_type wire      = f32_wire ? GGML_TYPE_F32 : q8_wire ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    const size_t    slot_offs = ggml_vk_comm_slot_off(comm->cap, round);
    const size_t    slot_size = ggml_row_size(wire, ne);
    ggml_tensor     slot      = *comm->slot;

    // A: own partial to own host slot, signal
    comm->rec_round[i].store(round, std::memory_order_relaxed);
    {
        vk_buffer & tbuf = ((ggml_backend_vk_buffer_context *) t->buffer->context)->dev_buffer;
        vk_buffer & hbuf = ((ggml_backend_vk_buffer_context *) d.host_buf[i]->context)->dev_buffer;
        const size_t toffs = vk_tensor_offset(t) + t->view_offs;
        vk_context c = ggml_vk_get_compute_ctx(ctx);
        ggml_vk_sync_buffers(ctx, c);
        ggml_vk_comm_ts(d, c, round, 0);
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            c->s->buffer->buf.fillBuffer(hbuf->buffer, slot_offs, GGML_PAD(slot_size, 4), 0);
            c->s->buffer->buf.fillBuffer(tbuf->buffer, toffs, nbytes, 0);
        } else if (wire == GGML_TYPE_F32) {
            if (!(sp_mode && (local_spin_exp & 4))) ggml_vk_buffer_copy_async(c, hbuf, slot_offs, tbuf, toffs, nbytes);
        } else if (!(local_comm_exp & 1)) {
            // f16 straight into host memory is 2-byte stores over PCIe, much slower than converting in
            // VRAM and letting the copy engine move whole lines (experiment, LOCAL_VK_COMM_STAGE=1; neutral while the IOMMU translates host DMA)
            // default: staged for the q8 wire (its quantize shader writes scattered 34-byte blocks), direct for f16
            static const int local_comm_stage = [] { const char * e = getenv("LOCAL_VK_COMM_STAGE"); return e ? atoi(e) : -1; }();
            const bool use_stage = local_comm_stage >= 0 ? local_comm_stage != 0 : wire == GGML_TYPE_Q8_0;
            if (use_stage && d.stage) {
                ggml_tensor st = slot;
                ggml_vk_comm_set_slot(&st, t, wire, d.stage, 0);
                ggml_vk_cpy(ctx, c, t, &st);
                ggml_vk_sync_buffers(ctx, c);
                vk_buffer & sbuf = ((ggml_backend_vk_buffer_context *) d.stage->context)->dev_buffer;
                if (!(local_comm_exp & 4)) {
                    ggml_vk_buffer_copy_async(c, hbuf, slot_offs, sbuf, 0, slot_size);
                }
            } else {
                ggml_vk_comm_set_slot(&slot, t, wire, d.host_buf[i], slot_offs);
                ggml_vk_cpy(ctx, c, t, &slot);
            }
        }
        ggml_vk_comm_ts(d, c, round, 1);
        if (tl_mode) {
            ggml_vk_comm_barrier_to_host(c);
            ggml_vk_comm_submit(ctx, c, d.tl, round);
        } else if (sp_mode) {
            // data reaches host memory before the flag write starts
            const vk::MemoryBarrier data_out{ vk::AccessFlagBits::eTransferWrite | vk::AccessFlagBits::eShaderWrite,
                                              vk::AccessFlagBits::eHostRead | vk::AccessFlagBits::eTransferWrite };
            c->s->buffer->buf.pipelineBarrier(c->p->q->stage_flags, vk::PipelineStageFlagBits::eHost | vk::PipelineStageFlagBits::eTransfer, {}, { data_out }, {}, {});
            static const bool local_spin_wsplit = std::getenv("LOCAL_VK_SPIN_WSPLIT") != nullptr;
            if (local_spin_wsplit) {
                // end of submission writes L2 back, the flag goes in the next one
                ggml_vk_comm_submit(ctx, c, {}, 0);
                c = ggml_vk_get_compute_ctx(ctx);
            }
            static const uint32_t local_fill_lines = [] { const char * e = std::getenv("LOCAL_VK_SPIN_FILL_LINES"); return e ? (uint32_t) atoi(e) : (uint32_t) LOCAL_FLAG_USE_LINES; }();
            if (!(local_spin_exp & 8)) c->s->buffer->buf.fillBuffer(hbuf->buffer, ggml_vk_comm_flag_off(comm->cap, round), local_fill_lines * LOCAL_FLAG_LINE, (uint32_t) round);
            if (!(local_spin_exp & 8) && !(local_spin_exp & 16)) ggml_vk_comm_barrier_to_host(c);
            static const bool local_spin_submit = [] { const char * e = std::getenv("LOCAL_VK_SPIN_SUBMIT"); return e != nullptr && atoi(e) != 0; }();
            // one host thread drives all devices: a later CPU wait on this device must not hold back a peer's A
            // the job's last round is recorded after the graph's final submit; between jobs the host may block
            // in a per-device synchronize while a peer still holds this signal unsubmitted, so send it now
            ggml_vk_comm_ts(d, c, round, 2);
            if (local_spin_submit || !comm->threaded || round == comm->round) {
                ggml_vk_comm_submit(ctx, c, {}, 0);
            }
        } else if (ev_mode) {
            // data reaches host memory before the flag
            ggml_vk_comm_barrier_to_host(c);
            c->s->buffer->buf.fillBuffer(hbuf->buffer, ggml_vk_comm_flag_off(comm->cap, round), sp_mode ? LOCAL_FLAG_PAGE : 4, (uint32_t) round);
            ggml_vk_comm_barrier_to_host(c);
            static const bool local_no_submit = std::getenv("LOCAL_VK_EVENT_NO_SUBMIT") != nullptr;
            if (!local_no_submit) {
                ggml_vk_comm_submit(ctx, c, {}, 0);
            }
        } else {
            ggml_vk_comm_sfd_signal(comm, i, c);
        }
    }
    if (!tl_mode && !sp_mode && comm->arrived[round & 1].fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
        comm->arrived[round & 1].store(0, std::memory_order_relaxed);
        ggml_vk_comm_notify_submitted(comm, round);
    }

    static const bool local_ev_log = std::getenv("LOCAL_VK_EVENT_LOG") != nullptr;
    if (local_ev_log) {
        GGML_LOG_INFO("EVT dev%zu A round %llu nbytes %zu wire %s compute=%d\n", i, (unsigned long long) round, nbytes, ggml_type_name(wire), (int) !!(t->flags & GGML_TENSOR_FLAG_COMPUTE));
    }
    // B: wait on peers, add their slots
    static const bool local_no_fused_reduce = std::getenv("LOCAL_VK_NO_FUSED_REDUCE") != nullptr;
    static const bool local_no_fold = std::getenv("LOCAL_VK_NO_FOLD_RESIDUAL") != nullptr;
    vk_pipeline fused = (!local_no_fused_reduce && wire == GGML_TYPE_F32 && ctx->device->multi_add && !ctx->device->disable_fusion &&
                         n + 1 <= MAX_PARAMETER_COUNT) ? ctx->device->pipeline_multi_add[n - 2] : nullptr;
    const bool fold = fused && !local_no_fold && ctx->local_tail == t && ctx->local_tail_uses == 1;
    // the deferred consumer does the spin wait itself (fold_norm) or emits it first, saving a dispatch and a barrier
    static const bool local_fold_wait = [] { const char * e = std::getenv("LOCAL_VK_FOLD_WAIT"); return e == nullptr || atoi(e) != 0; }();
    const bool defer_wait = sp_mode && fold && local_fold_wait && !(local_spin_exp & 1) && n - 1 <= 3;
    vk_context c = ggml_vk_get_compute_ctx(ctx);
    ctx->do_add_rms_partials = false;
    static const bool local_event_wait = std::getenv("LOCAL_VK_EVENT_WAIT") != nullptr;
    if (tl_mode) {
        for (size_t j = 0; j < n; j++) {
            if (j != i) {
                c->s->wait_semaphores.push_back({ d.tl_imp[j], round });
            }
        }
        ggml_vk_comm_barrier_from_host(c);
    } else if (sp_mode) {
        vk_pipeline & pw = ctx->device->pipeline_comm_wait;
        ggml_pipeline_request_descriptor_sets(ctx, pw, 1);
        vk_subbuffer fl[8];
        uint32_t np = 0;
        for (size_t j = 0; j < n; j++) {
            if (j != i) {
                vk_buffer & pb = ((ggml_backend_vk_buffer_context *) comm->devs[j].host_buf[i]->context)->dev_buffer;
                fl[np++] = { pb, ggml_vk_comm_flag_off(comm->cap, round), LOCAL_FLAG_PAGE };
            }
        }
        for (uint32_t k = np; k < 8; k++) {
            fl[k] = fl[0];
        }
        {
            vk_buffer & ob = ((ggml_backend_vk_buffer_context *) d.host_buf[i]->context)->dev_buffer;
            fl[7] = { ob, ggml_vk_comm_dbg_off(comm->cap), 128 };
        }
        // bound in realtime clock ticks (27 MHz crystal on Tonga). Normal waits are tens of microseconds; the
        // bound only turns a dead or hung peer into an abort. Default 30 s: a first-use pipeline compile on
        // one device's host thread can hold its peers for seconds.
        static const uint32_t local_spin_limit = [] {
            const char * e = std::getenv("LOCAL_VK_SPIN_LIMIT_US");
            const uint64_t us = e ? (uint64_t) atoll(e) : 30000000u;
            return (uint32_t) std::min<uint64_t>(us * 27, 0xF0000000u);
        }();
        vk_buffer & sb = ((ggml_backend_vk_buffer_context *) d.scratch->context)->dev_buffer;
        const vk_subbuffer sweep = { sb, 0, ggml_backend_buffer_get_size(d.scratch) };
        static const uint32_t local_poll_lines = [] { const char * e = std::getenv("LOCAL_VK_SPIN_FILL_LINES"); return e ? (uint32_t) atoi(e) : (uint32_t) LOCAL_FLAG_USE_LINES; }();
        struct { uint32_t round, n, limit, stride, lines, sweep; } pc = { (uint32_t) round, np, local_spin_limit, (uint32_t) (LOCAL_FLAG_LINE / 4), local_poll_lines, (uint32_t) (sweep.size / 16) };
        if (defer_wait) {
            static const bool local_log = getenv("LOCAL_VK_FOLD_WAIT_LOG") != nullptr;
            if (local_log) {
                fprintf(stderr, "DEFERWAIT dev%zu round %llu\n", i, (unsigned long long) round);
            }
            for (uint32_t k = 0; k < 8; k++) {
                ctx->local_pending_wait_buf[k] = fl[k];
            }
            ctx->local_pending_wait_buf[8] = sweep;
            memcpy(ctx->local_pending_wait_pc, &pc, sizeof(pc));
        } else if (!(local_spin_exp & 1)) {
            ggml_vk_dispatch_pipeline(ctx, c, pw, { fl[0], fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7], sweep }, pc, { 1, 1, 1 });
        }
        static const bool local_spin_split = std::getenv("LOCAL_VK_SPIN_SPLIT") != nullptr;
        if (local_spin_split) {
            ggml_vk_comm_submit(ctx, c, {}, 0);
            c = ggml_vk_get_compute_ctx(ctx);
        }
        // everything after the wait sees the peers' data
        const vk::MemoryBarrier data_in{ vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eShaderWrite | vk::AccessFlagBits::eShaderRead,
                                         vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead };
        if (!defer_wait && !(local_spin_exp & 2)) c->s->buffer->buf.pipelineBarrier(vk::PipelineStageFlagBits::eHost | c->p->q->stage_flags, c->p->q->stage_flags, {}, { data_in }, {}, {});
    } else if (ev_mode && local_event_wait) {
        // host-set events are not seen reliably by GCN3 when the wait started earlier
        const vk::MemoryBarrier from_host{ vk::AccessFlagBits::eHostWrite, vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eTransferRead };
        c->s->buffer->buf.waitEvents({ d.ev[round & 1] }, vk::PipelineStageFlagBits::eHost | c->p->q->stage_flags, c->p->q->stage_flags, { from_host }, {}, {});
        c->s->buffer->buf.resetEvent(d.ev[round & 1], c->p->q->stage_flags);
    } else if (ev_mode) {
        // submit of B is held on the host until all flags of this round are seen
        c->s->ready       = &comm->imported;
        c->s->ready_value = round;
        ggml_vk_comm_barrier_from_host(c);
    } else {
        ggml_vk_comm_sfd_wait(comm, i, c, round);
    }
    ggml_vk_sync_buffers(ctx, c);
    ggml_vk_comm_ts(d, c, round, 3);

    vk_pipeline q8_sum = (q8_wire && ctx->device->multi_add && n + 1 <= MAX_PARAMETER_COUNT) ? ctx->device->pipeline_multi_add[n - 2] : nullptr;
    if (q8_wire) {
        // dequantize each peer's q8_0 slot into VRAM, then one f32 multi_add (or plain adds)
        ggml_tensor peers[MAX_PARAMETER_COUNT];
        const ggml_tensor * srcs[MAX_PARAMETER_COUNT];
        uint32_t k = 0;
        srcs[k++] = t;
        for (size_t j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            ggml_tensor q = *comm->slot;
            ggml_vk_comm_set_slot(&q, t, wire, comm->devs[j].host_buf[i], slot_offs);
            peers[k] = *comm->slot;
            ggml_vk_comm_set_slot(&peers[k], t, GGML_TYPE_F32, d.peer_f32, (k - 1) * comm->cap);
            ggml_vk_cpy(ctx, c, &q, &peers[k]);
            srcs[k] = &peers[k];
            k++;
        }
        ggml_vk_sync_buffers(ctx, c);
        if (q8_sum) {
            srcs[k] = t;
            ggml_vk_multi_add_tensors(ctx, c, q8_sum, srcs, k);
        } else {
            for (uint32_t m = 1; m < k; m++) {
                ggml_vk_add(ctx, c, t, &peers[m], t);
                ggml_vk_sync_buffers(ctx, c);
            }
        }
        ggml_vk_sync_buffers(ctx, c);
    } else if (fold) {
        GGML_ASSERT(!ctx->local_pending_active);
        uint32_t k = 1;
        for (size_t j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            ggml_tensor & sl = ctx->local_pending_slots[k++];
            sl = *comm->slot;
            ggml_vk_comm_set_slot(&sl, t, wire, comm->devs[j].host_buf[i], slot_offs);
        }
        ctx->local_pending_n = k;
        ctx->local_pending_own = t;
        ctx->local_pending_active = true;
        ctx->local_pending_hold = false;
        ctx->local_pending_fold_node = nullptr;
        ctx->local_pending_wait = defer_wait;
    } else if (fused) {
        ggml_tensor peers[MAX_PARAMETER_COUNT];
        const ggml_tensor * srcs[MAX_PARAMETER_COUNT];
        uint32_t k = 0;
        srcs[k++] = t;
        for (size_t j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            peers[k] = *comm->slot;
            ggml_vk_comm_set_slot(&peers[k], t, wire, comm->devs[j].host_buf[i], slot_offs);
            srcs[k] = &peers[k];
            k++;
        }
        srcs[k] = t;
        ggml_vk_multi_add_tensors(ctx, c, fused, srcs, k);
        ggml_vk_sync_buffers(ctx, c);
    } else if (!(local_comm_exp & 2)) {
        for (size_t j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            ggml_vk_comm_set_slot(&slot, t, wire, comm->devs[j].host_buf[i], slot_offs);
            ggml_vk_add(ctx, c, t, &slot, t);
            ggml_vk_sync_buffers(ctx, c);
        }
    }
    ggml_vk_comm_ts(d, c, round, 4);
    if (wire != GGML_TYPE_F32 || vk_perf_logger_enabled) {
        ggml_vk_comm_submit(ctx, c, {}, 0);
    }
    return true;
}
#endif

// Probes the power/energy APIs of the local GPU: which are supported, how
// often each one updates, and whether the hardware energy counter agrees
// with the nvmlDeviceGetSamples trapezoid.
//
// Build:  nvcc -O2 -lnvidia-ml -o probe_energy_counter probe_energy_counter.cu
// Run:    ./probe_energy_counter [cuda_device_id]

#include <nvml.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>

static void check_nvml(nvmlReturn_t r, const char *msg) {
    if (r != NVML_SUCCESS) {
        fprintf(stderr, "NVML error: %s: %s\n", msg, nvmlErrorString(r));
        exit(1);
    }
}

// Read a field-values entry (e.g. NVML_FI_DEV_POWER_AVERAGE = 185).
static bool field_read(nvmlDevice_t dev, unsigned int id, unsigned int *out) {
    nvmlFieldValue_t fv = {};
    fv.fieldId = id;
    if (nvmlDeviceGetFieldValues(dev, 1, &fv) != NVML_SUCCESS ||
        fv.nvmlReturn != NVML_SUCCESS)
        return false;
    *out = fv.value.uiVal;
    return true;
}

// Drain a GetSamples buffer from the beginning; returns its newest timestamp.
static unsigned long long drain_ts(nvmlDevice_t dev, nvmlSamplingType_t type) {
    nvmlValueType_t vt; unsigned int cnt = 0;
    nvmlDeviceGetSamples(dev, type, 0, &vt, &cnt, nullptr);
    if (cnt == 0) return 0;
    std::vector<nvmlSample_t> buf(cnt);
    nvmlDeviceGetSamples(dev, type, 0, &vt, &cnt, buf.data());
    unsigned long long mx = 0;
    for (auto &s : buf) if (s.timeStamp > mx) mx = s.timeStamp;
    return mx;
}

// Update-rate tracker: records ms between consecutive observed changes.
struct Rate {
    long long last_us = 0;
    std::vector<double> dts_ms;
};
static void on_change(Rate &r, long long now_us) {
    if (r.last_us > 0) r.dts_ms.push_back((now_us - r.last_us) / 1000.0);
    r.last_us = now_us;
}
static void print_rate(const char *name, const Rate &r) {
    if (r.dts_ms.empty()) {
        printf("  [%s] no changes observed\n", name);
        return;
    }
    double sum = 0, mn = r.dts_ms[0], mx = r.dts_ms[0];
    for (double d : r.dts_ms) {
        sum += d;
        if (d < mn) mn = d;
        if (d > mx) mx = d;
    }
    double mean = sum / r.dts_ms.size();
    printf("  [%s] %3zu change(s)  dt: mean=%.1f  min=%.1f  max=%.1f ms  -> ~%.0f Hz\n",
           name, r.dts_ms.size(), mean, mn, mx, 1000.0 / mean);
}

int main(int argc, char **argv) {
    int device_id = (argc > 1) ? atoi(argv[1]) : 0;
    check_nvml(nvmlInit(), "nvmlInit");

    // Resolve via PCI bus ID: the one identifier CUDA and NVML agree on.
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed\n");
        return 1;
    }
    char pci_id[32];
    snprintf(pci_id, sizeof(pci_id), "%08x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    printf("Device: %s (PCI %s)\n", prop.name, pci_id);
    nvmlDevice_t dev;
    check_nvml(nvmlDeviceGetHandleByPciBusId(pci_id, &dev), "GetHandleByPciBusId");

    // -- 1. Energy counter support -----------------------------------------
    unsigned long long e_prev = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(dev, &e_prev) != NVML_SUCCESS) {
        printf("getTotalEnergyConsumption NOT SUPPORTED -- nothing to compare\n");
        nvmlShutdown();
        return 0;
    }
    printf("getTotalEnergyConsumption supported (initially %llu mJ)\n\n", e_prev);

    // -- 2. Update rates of the power APIs (poll ~1 ms for 2 s) ------------
    unsigned int tmp = 0;
    bool has_avg = field_read(dev, 185, &tmp);   // NVML_FI_DEV_POWER_AVERAGE
    bool has_ins = field_read(dev, 186, &tmp);   // NVML_FI_DEV_POWER_INSTANT
    bool has_mod = drain_ts(dev, NVML_MODULE_POWER_SAMPLES) > 0;
    printf("POWER_AVERAGE: %s  POWER_INSTANT: %s  MODULE_POWER_SAMPLES: %s\n\n",
           has_avg ? "supported" : "not supported",
           has_ins ? "supported" : "not supported",
           has_mod ? "supported" : "not supported");

    unsigned long long smp_ts = drain_ts(dev, NVML_TOTAL_POWER_SAMPLES);
    unsigned long long mod_ts = has_mod ? drain_ts(dev, NVML_MODULE_POWER_SAMPLES) : 0;
    unsigned int avg_prev = 0, pwr_prev = 0, ins_prev = 0;
    field_read(dev, 185, &avg_prev);
    nvmlDeviceGetPowerUsage(dev, &pwr_prev);
    if (has_ins) field_read(dev, 186, &ins_prev);

    Rate r_ctr, r_avg, r_pwr, r_ins, r_smp, r_mod;
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < 2000; i++) {
        long long us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - t0).count();

        unsigned long long e;
        nvmlDeviceGetTotalEnergyConsumption(dev, &e);
        if (e != e_prev) { on_change(r_ctr, us); e_prev = e; }

        unsigned int v;
        if (has_avg && field_read(dev, 185, &v) && v != avg_prev) {
            on_change(r_avg, us); avg_prev = v;
        }
        if (nvmlDeviceGetPowerUsage(dev, &v) == NVML_SUCCESS && v != pwr_prev) {
            on_change(r_pwr, us); pwr_prev = v;
        }
        if (has_ins && field_read(dev, 186, &v) && v != ins_prev) {
            on_change(r_ins, us); ins_prev = v;
        }

        // Each new GetSamples entry is one update of the sampled API.
        nvmlValueType_t vt; unsigned int cnt = 0;
        nvmlDeviceGetSamples(dev, NVML_TOTAL_POWER_SAMPLES, smp_ts, &vt, &cnt, nullptr);
        if (cnt) {
            std::vector<nvmlSample_t> b(cnt);
            nvmlDeviceGetSamples(dev, NVML_TOTAL_POWER_SAMPLES, smp_ts, &vt, &cnt, b.data());
            for (auto &s : b)
                if (s.timeStamp > smp_ts) { on_change(r_smp, us); smp_ts = s.timeStamp; }
        }
        if (has_mod) {
            cnt = 0;
            nvmlDeviceGetSamples(dev, NVML_MODULE_POWER_SAMPLES, mod_ts, &vt, &cnt, nullptr);
            if (cnt) {
                std::vector<nvmlSample_t> b(cnt);
                nvmlDeviceGetSamples(dev, NVML_MODULE_POWER_SAMPLES, mod_ts, &vt, &cnt, b.data());
                for (auto &s : b)
                    if (s.timeStamp > mod_ts) { on_change(r_mod, us); mod_ts = s.timeStamp; }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    printf("Update rates over 2 s (poll interval 1 ms):\n");
    print_rate("CTR", r_ctr);   // getTotalEnergyConsumption
    print_rate("AVG", r_avg);   // POWER_AVERAGE field
    print_rate("PWR", r_pwr);   // getPowerUsage
    print_rate("INS", r_ins);   // POWER_INSTANT field
    print_rate("SMP", r_smp);   // GetSamples(TOTAL_POWER)
    print_rate("MOD", r_mod);   // GetSamples(MODULE_POWER)

    // -- 3. Counter bracket vs GetSamples trapezoid over a 200 ms window ---
    unsigned long long ts0 = drain_ts(dev, NVML_TOTAL_POWER_SAMPLES);
    unsigned long long ea, eb;
    nvmlDeviceGetTotalEnergyConsumption(dev, &ea);
    auto tb = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    nvmlDeviceGetTotalEnergyConsumption(dev, &eb);
    double wall_s = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - tb).count();

    nvmlValueType_t vt; unsigned int cnt = 0;
    nvmlDeviceGetSamples(dev, NVML_TOTAL_POWER_SAMPLES, ts0, &vt, &cnt, nullptr);
    std::vector<nvmlSample_t> s(cnt ? cnt : 1);
    if (cnt)
        nvmlDeviceGetSamples(dev, NVML_TOTAL_POWER_SAMPLES, ts0, &vt, &cnt, s.data());

    double trap_J = 0.0;
    for (size_t i = 0; i + 1 < s.size(); i++) {
        if (s[i + 1].timeStamp <= s[i].timeStamp) continue;
        double p0 = s[i].sampleValue.uiVal / 1000.0;      // mW -> W
        double p1 = s[i + 1].sampleValue.uiVal / 1000.0;
        trap_J += (p0 + p1) * 0.5 * (s[i + 1].timeStamp - s[i].timeStamp) * 1e-6;
    }
    double ctr_J = (double)(eb - ea) / 1000.0;
    printf("\n200 ms idle window: counter %.4f J (%.1f W)  vs  trapezoid %.4f J"
           "  (%u samples)  ratio %.4f\n",
           ctr_J, ctr_J / wall_s, trap_J, cnt,
           trap_J > 0.0 ? ctr_J / trap_J : 0.0);

    nvmlShutdown();
    return 0;
}

#ifndef METRICS_H
#define METRICS_H

// GPU telemetry monitoring via NVML.
// Everything lives inside #ifdef METRICS so the header is always safe to
// include but compiles to nothing when metrics are disabled.

#ifdef METRICS

#include <cuda_runtime.h>
#include <nvml.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>

// Averaged metrics returned by GpuMonitor::averages().
struct MetricsAvg {
  double power;      // W
  double clock;      // MHz
  double temp;       // C
  double gpu_util;   // %
  double mem_util;   // %
  double energy_J;   // J   (trapezoidal integral over the run)
  double mem_power;  // W   (HBM rail; ~1s trailing average, 0 if unsupported)
};

// A [start, end) window in host-epoch µs timestamps.
struct TimeWindow {
  unsigned long long start_us;
  unsigned long long end_us;
};

// NVML wrapper that samples power/clock/temp/util on a background thread
// (~50 ms) and keeps the timestamped sample history for per-window energy
// and clock queries.  Usage:
//
//   GpuMonitor mon;
//   if (!mon.init(device_id, gpu_id)) return 1;
//   mon.start();          // resets accumulators, spawns thread
//   ... run kernels ...
//   mon.stop();           // joins thread
//   MetricsAvg a = mon.averages();
//   mon.shutdown();       // tear down backend
//
// start()/stop() may be called repeatedly.
struct GpuMonitor {
  // Accumulators (written only by the sampling thread)
  double totalPower;
  double totalClock;
  double totalTemp;
  unsigned long long totalGpuUtil;
  unsigned long long totalMemUtil;
  int poll_samples;
  int power_samples;
  double totalEnergy_J;
  double totalMemPower;
  int mempower_samples;

  // Thread control
  std::atomic<bool> _active;
  std::thread _worker;
  mutable std::mutex _mtx;   // guards the sample history below

  // Backend handle
  nvmlDevice_t _nvml_dev;

  // Sample history (guarded by _mtx)
  unsigned long long _lastPowerTs;
  std::vector<nvmlSample_t> _all_samples;                            // timestamped power history
  std::vector<std::pair<unsigned long long, double>> _clock_samples;   // (ts_us, MHz)
  bool _mempower_supported;
  std::vector<std::pair<unsigned long long, double>> _mempower_samples; // (ts_us, W)

  // Initialize the monitoring backend. Returns true on success.
  inline bool init(int device_id, int gpu_id)
  {
    nvmlReturn_t r = nvmlInit();
    if (r != NVML_SUCCESS) {
      std::cerr << "Failed to init NVML: " << nvmlErrorString(r) << std::endl;
      return false;
    }
    // Resolve via PCI bus ID: NVML must see the same GPU CUDA is using,
    // even when CUDA_VISIBLE_DEVICES remaps indices.
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
      std::cerr << "Failed to get CUDA device properties for device " << device_id << std::endl;
      return false;
    }
    char pci_id[32];
    snprintf(pci_id, sizeof(pci_id), "%08x:%02x:%02x.0",
             prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    r = nvmlDeviceGetHandleByPciBusId(pci_id, &_nvml_dev);
    if (r != NVML_SUCCESS) {
      std::cerr << "Failed to get NVML device by PCI ID " << pci_id
                << ": " << nvmlErrorString(r) << std::endl;
      return false;
    }
    _lastPowerTs = 0;
    (void)gpu_id;
    // Probe memory-rail support once; the #if keeps pre-12.x nvml.h compiling.
    _mempower_supported = false;
#ifdef NVML_FI_DEV_POWER_AVERAGE
    {
      nvmlFieldValue_t v = {};
      v.fieldId = NVML_FI_DEV_POWER_AVERAGE;
      v.scopeId = NVML_POWER_SCOPE_MEMORY;
      if (nvmlDeviceGetFieldValues(_nvml_dev, 1, &v) == NVML_SUCCESS &&
          v.nvmlReturn == NVML_SUCCESS)
        _mempower_supported = true;
    }
#endif
    return true;
  }

  // Reset accumulators and start the sampling thread.
  inline void start()
  {
    totalEnergy_J  = 0.0;
    totalPower     = 0.0;
    power_samples  = 0;
    totalClock     = 0.0;
    totalTemp      = 0.0;
    totalGpuUtil   = 0;
    totalMemUtil   = 0;
    poll_samples   = 0;
    totalMemPower    = 0.0;
    mempower_samples = 0;
    _mempower_samples.clear();
    {
      nvmlValueType_t vt; unsigned int cnt = 0;
      if (nvmlDeviceGetSamples(_nvml_dev, NVML_TOTAL_POWER_SAMPLES,
                               0, &vt, &cnt, nullptr) == NVML_SUCCESS && cnt > 0) {
        std::vector<nvmlSample_t> tmp(cnt);
        if (nvmlDeviceGetSamples(_nvml_dev, NVML_TOTAL_POWER_SAMPLES,
                                 0, &vt, &cnt, tmp.data()) == NVML_SUCCESS && cnt > 0)
          _lastPowerTs = tmp[cnt - 1].timeStamp;
        else
          _lastPowerTs = 0;
      } else {
        _lastPowerTs = 0;
      }
    }
    _all_samples.clear();
    _clock_samples.clear();
    _active.store(true);

    _worker = std::thread([this]() {
      while (_active.load()) {
        {
          std::lock_guard<std::mutex> lk(_mtx);
          _drain_power_samples();
          _poll_mempower();

          unsigned int clock_MHz = 0, temp_C = 0;
          nvmlUtilization_t util = {};
          if (nvmlDeviceGetClockInfo(_nvml_dev, NVML_CLOCK_GRAPHICS, &clock_MHz) == NVML_SUCCESS) {
            totalClock += clock_MHz;
            unsigned long long ts_us = (unsigned long long)
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
            _clock_samples.push_back({ts_us, (double)clock_MHz});
          }
          if (nvmlDeviceGetTemperature(_nvml_dev, NVML_TEMPERATURE_GPU, &temp_C) == NVML_SUCCESS)
            totalTemp += temp_C;
          if (nvmlDeviceGetUtilizationRates(_nvml_dev, &util) == NVML_SUCCESS) {
            totalGpuUtil += util.gpu;
            totalMemUtil += util.memory;
          }
          poll_samples++;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    });
  }

  // Stop the sampling thread and drain tail power samples.
  inline void stop()
  {
    _active.store(false);
    if (_worker.joinable()) _worker.join();
    std::lock_guard<std::mutex> lk(_mtx);
    _drain_power_samples();
  }

  // Compute averaged metrics from accumulated samples.
  inline MetricsAvg averages() const
  {
    MetricsAvg avg = {};
    if (power_samples > 0) {
      avg.power    = totalPower / power_samples;
      avg.energy_J = totalEnergy_J;
    }
    if (poll_samples > 0) {
      avg.clock    = totalClock    / poll_samples;
      avg.temp     = totalTemp     / poll_samples;
      avg.gpu_util = (double)totalGpuUtil / poll_samples;
      avg.mem_util = (double)totalMemUtil / poll_samples;
    }
    if (mempower_samples > 0)
      avg.mem_power = totalMemPower / mempower_samples;
    return avg;
  }

  // "SM Util" label for the summary lines.
  static inline const char *util_label()
  {
    return "SM Util";
  }

  // Shut down the monitoring backend.
  inline void shutdown()
  {
    nvmlShutdown();
  }

  // -----------------------------------------------------------------------
  // Per-window queries over the sample history (timestamps in host-epoch µs)
  // -----------------------------------------------------------------------

  // Flush NVML ring buffer from the main thread (catches tail samples).
  inline void drain()
  {
    std::lock_guard<std::mutex> lk(_mtx);
    _drain_power_samples();
  }

  // Hardware energy counter bracket. Returns -1 if unsupported.
  inline long long energy_counter_mj() const
  {
    unsigned long long mj = 0;
    if (nvmlDeviceGetTotalEnergyConsumption(_nvml_dev, &mj) == NVML_SUCCESS)
      return (long long)mj;
    return -1LL;
  }

  // Energy (J) during the windows, linearly interpolating power samples at
  // the boundaries (handles samples straddling an edge).
  inline double energy_in_windows(const std::vector<TimeWindow>& windows) const
  {
    if (_all_samples.size() < 2 || windows.empty()) return 0.0;

    double energy = 0.0;
    for (size_t i = 0; i + 1 < _all_samples.size(); i++) {
      unsigned long long ts0 = _all_samples[i].timeStamp;
      unsigned long long ts1 = _all_samples[i + 1].timeStamp;
      if (ts1 <= ts0) continue;
      double p0 = _all_samples[i].sampleValue.uiVal     / 1000.0;  // mW → W
      double p1 = _all_samples[i + 1].sampleValue.uiVal / 1000.0;

      for (const auto& w : windows) {
        unsigned long long clip_s = std::max(ts0, w.start_us);
        unsigned long long clip_e = std::min(ts1, w.end_us);
        if (clip_e <= clip_s) continue;
        // Linear interpolation of power at the clipped endpoints.
        double span  = (double)(ts1 - ts0);
        double pc_s  = p0 + (p1 - p0) * (double)(clip_s - ts0) / span;
        double pc_e  = p0 + (p1 - p0) * (double)(clip_e - ts0) / span;
        energy += (pc_s + pc_e) * 0.5 * (double)(clip_e - clip_s) * 1e-6;
      }
    }
    return energy;
  }

  // Single-window variant, safe to call while the monitor is running.
  inline double energy_in_window_locked(TimeWindow w) const
  {
    std::lock_guard<std::mutex> lk(_mtx);
    return energy_in_windows({w});
  }

  // Mean SM clock (MHz) over a single time window.
  inline double avg_clock_in_window(unsigned long long start_us, unsigned long long end_us) const
  {
    std::lock_guard<std::mutex> lk(_mtx);
    double sum = 0.0; int n = 0;
    for (const auto& s : _clock_samples)
      if (s.first >= start_us && s.first < end_us) { sum += s.second; n++; }
    return n > 0 ? sum / n : 0.0;
  }

  // SM clock (MHz) from the background sample nearest to ts_us.
  inline unsigned int clock_near_us(unsigned long long ts_us) const
  {
    std::lock_guard<std::mutex> lk(_mtx);
    if (_clock_samples.empty()) return 0;
    unsigned long long best_diff = ULLONG_MAX;
    double best = 0.0;
    for (const auto& s : _clock_samples) {
      unsigned long long diff = (s.first >= ts_us) ? s.first - ts_us : ts_us - s.first;
      if (diff < best_diff) { best_diff = diff; best = s.second; }
    }
    return (unsigned int)best;
  }

  // Mean SM clock (MHz) during the given windows.
  inline double avg_clock_in_windows(const std::vector<TimeWindow>& windows) const
  {
    std::lock_guard<std::mutex> lk(_mtx);
    double sum = 0.0; int n = 0;
    for (const auto& s : _clock_samples) {
      for (const auto& w : windows) {
        if (s.first >= w.start_us && s.first < w.end_us) {
          sum += s.second; n++; break;
        }
      }
    }
    return n > 0 ? sum / n : 0.0;
  }

  // Mean HBM-rail power during the windows (~1s smoothing: only meaningful
  // for windows >= 1s). 0 if unsupported.
  inline double avg_mempower_in_windows(const std::vector<TimeWindow>& windows) const
  {
    std::lock_guard<std::mutex> lk(_mtx);
    double sum = 0.0; int n = 0;
    for (const auto& s : _mempower_samples) {
      for (const auto& w : windows) {
        if (s.first >= w.start_us && s.first < w.end_us) {
          sum += s.second; n++; break;
        }
      }
    }
    return n > 0 ? sum / n : 0.0;
  }

  // Population stddev of power samples inside the windows: how stable the
  // draw was (high stddev = DVFS swings or cold-start ramp-up).
  inline double power_stddev_in_windows(const std::vector<TimeWindow>& windows) const
  {
    if (_all_samples.empty() || windows.empty()) return 0.0;

    std::vector<double> vals;
    for (const auto& s : _all_samples) {
      for (const auto& w : windows) {
        if (s.timeStamp >= w.start_us && s.timeStamp < w.end_us) {
          vals.push_back(s.sampleValue.uiVal / 1000.0);
          break;
        }
      }
    }
    if (vals.size() < 2) return 0.0;

    double mean = 0.0;
    for (double v : vals) mean += v;
    mean /= (double)vals.size();

    double var = 0.0;
    for (double v : vals) var += (v - mean) * (v - mean);
    return std::sqrt(var / (double)vals.size());
  }

  // Raw sample history as CSV rows (kind,,,host_us,value,unit). All
  // timestamps are Unix-epoch µs, so they align with the event timeline.
  inline void write_trace_rows(std::ostream& os) const
  {
    for (const auto& s : _all_samples)
      os << "power,,," << s.timeStamp << "," << (s.sampleValue.uiVal) << ",mW\n";
    for (const auto& c : _clock_samples)
      os << "clock,,," << c.first << "," << c.second << ",MHz\n";
    for (const auto& m : _mempower_samples)
      os << "mempower,,," << m.first << "," << m.second << ",W\n";
  }

  // -----------------------------------------------------------------------
  // Internal: worker-thread sampling
  // -----------------------------------------------------------------------

  // Drain NVML power samples; trapezoidal-integrate into totalEnergy_J.
  inline void _drain_power_samples()
  {
    nvmlValueType_t valType;
    unsigned int count = 0;
    nvmlReturn_t r = nvmlDeviceGetSamples(_nvml_dev, NVML_TOTAL_POWER_SAMPLES,
                                           _lastPowerTs, &valType, &count, nullptr);
    if (r != NVML_SUCCESS || count == 0) return;

    std::vector<nvmlSample_t> buf(count);
    r = nvmlDeviceGetSamples(_nvml_dev, NVML_TOTAL_POWER_SAMPLES,
                              _lastPowerTs, &valType, &count, buf.data());
    if (r != NVML_SUCCESS || count == 0) return;

    double             prev_W  = _all_samples.empty() ? -1.0
                                  : _all_samples.back().sampleValue.uiVal / 1000.0;
    unsigned long long prev_ts = _lastPowerTs;

    for (unsigned int i = 0; i < count; i++) {
      double cur_W = buf[i].sampleValue.uiVal / 1000.0;  // mW → W
      totalPower += cur_W;
      power_samples++;
      if (prev_W >= 0.0 && buf[i].timeStamp > prev_ts) {
        double dt = (buf[i].timeStamp - prev_ts) * 1e-6;  // µs → s
        totalEnergy_J += (prev_W + cur_W) * 0.5 * dt;
      }
      prev_W  = cur_W;
      prev_ts = buf[i].timeStamp;
      _all_samples.push_back(buf[i]);
    }
    _lastPowerTs = buf[count - 1].timeStamp;
  }

  // Sample HBM-rail power (worker thread, under _mtx).
  inline void _poll_mempower()
  {
#ifdef NVML_FI_DEV_POWER_AVERAGE
    if (!_mempower_supported) return;
    nvmlFieldValue_t v = {};
    v.fieldId = NVML_FI_DEV_POWER_AVERAGE;
    v.scopeId = NVML_POWER_SCOPE_MEMORY;
    if (nvmlDeviceGetFieldValues(_nvml_dev, 1, &v) != NVML_SUCCESS ||
        v.nvmlReturn != NVML_SUCCESS)
      return;
    double W = (v.valueType == NVML_VALUE_TYPE_UNSIGNED_LONG_LONG)
                   ? v.value.ullVal / 1000.0
                   : v.value.uiVal / 1000.0;  // mW -> W
    unsigned long long ts_us = v.timestamp;   // epoch µs, same domain as clock rows
    if (ts_us == 0)
      ts_us = (unsigned long long)std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count();
    totalMemPower += W;
    mempower_samples++;
    _mempower_samples.push_back({ts_us, W});
#endif
  }
};

#endif // METRICS
#endif // METRICS_H

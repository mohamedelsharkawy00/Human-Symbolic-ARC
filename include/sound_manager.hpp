#pragma once

// Sound is Windows-only (WinMM waveOut). On non-MSVC builds (e.g. Kaggle gcc),
// compile a no-op sound_manager with the same public API so core call sites
// (play / play_sequence / clear_queue) need no changes. See KAGGLE_PORT_PLAN.txt.
#ifndef _MSC_VER

#include <vector>
#include <utility>

namespace arc {

class sound_manager {
public:
    static sound_manager& get() { static sound_manager i; return i; }
    void play(int, int) {}
    void play_sequence(const std::vector<std::pair<int, int>>&) {}
    void clear_queue() {}
    void set_muted(bool) {}
    bool is_muted() const { return false; }
};

}  // namespace arc

#else  // _MSC_VER : real Windows implementation

#include <windows.h>
#include <mmsystem.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace arc {

class sound_manager {
public:
    static sound_manager& get() {
        static sound_manager instance;
        return instance;
    }

    void play(int freq, int duration_ms) {
        if (m_muted.load(std::memory_order_relaxed)) return;
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            m_queue.push({freq, duration_ms});
        }
        m_cv.notify_one();
    }

    void play_sequence(const std::vector<std::pair<int, int>>& sequence) {
        if (m_muted.load(std::memory_order_relaxed)) return;
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            for (const auto& step : sequence) {
                m_queue.push(step);
            }
        }
        m_cv.notify_one();
    }

    // Mute gates new sounds at the play() entry points and drops anything
    // already queued so muting takes effect immediately. A tone mid-render
    // finishes (it's short, with a fade-out); nothing new follows it.
    void set_muted(bool muted) {
        m_muted.store(muted, std::memory_order_relaxed);
        if (muted) clear_queue();
    }
    bool is_muted() const { return m_muted.load(std::memory_order_relaxed); }

    void clear_queue() {
        {
            std::lock_guard<std::mutex> lock(m_mtx);
            while (!m_queue.empty()) m_queue.pop();
        }
    }

private:
    sound_manager() {
        WAVEFORMATEX wfx = {};
        wfx.wFormatTag = WAVE_FORMAT_PCM;
        wfx.nChannels = 1;
        wfx.nSamplesPerSec = 44100;
        wfx.wBitsPerSample = 16;
        wfx.nBlockAlign = (wfx.nChannels * wfx.wBitsPerSample) / 8;
        wfx.nAvgBytesPerSec = wfx.nSamplesPerSec * wfx.nBlockAlign;
        wfx.cbSize = 0;

        if (waveOutOpen(&m_hWaveOut, WAVE_MAPPER, &wfx, 0, 0, CALLBACK_NULL) != MMSYSERR_NOERROR) {
            m_hWaveOut = nullptr;
        }

        m_worker = std::jthread([this](std::stop_token st) {
            while (!st.stop_requested()) {
                std::pair<int, int> task;
                {
                    std::unique_lock<std::mutex> lock(m_mtx);
                    m_cv.wait(lock, st, [this] { return !m_queue.empty(); });
                    if (st.stop_requested()) break;
                    if (m_queue.empty()) continue;
                    task = m_queue.front();
                    m_queue.pop();
                }

                if (task.first > 0) {
                    generate_and_play(task.first, task.second);
                } else if (task.second > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(task.second));
                }
            }
        });
    }

    ~sound_manager() {
        m_worker.request_stop();
        m_cv.notify_all();
        if (m_worker.joinable()) {
            m_worker.join();
        }
        if (m_hWaveOut) {
            waveOutReset(m_hWaveOut);
            waveOutClose(m_hWaveOut);
        }
    }

    void generate_and_play(int freq, int duration_ms) {
        if (!m_hWaveOut) return;

        const int sample_rate = 44100;
        const int num_samples = (sample_rate * duration_ms) / 1000;
        std::vector<int16_t> buffer(num_samples);

        const double pi = 3.14159265358979323846;
        const double freq_step = 2.0 * pi * freq / sample_rate;
        
        // Envelope settings: 10ms fade in/out to prevent clicking
        const int envelope_samples = (sample_rate * 10) / 1000;
        const int actual_envelope = std::min(envelope_samples, num_samples / 2);

        for (int i = 0; i < num_samples; ++i) {
            double amplitude = 0.5; // Base amplitude (0.0 to 1.0, using half for safety)
            
            // Apply linear envelope
            if (i < actual_envelope) {
                amplitude *= (double)i / actual_envelope;
            } else if (i > num_samples - actual_envelope) {
                amplitude *= (double)(num_samples - i) / actual_envelope;
            }

            double val = std::sin(i * freq_step) * amplitude;
            buffer[i] = static_cast<int16_t>(val * 32767.0);
        }

        WAVEHDR header = {};
        header.lpData = (LPSTR)buffer.data();
        header.dwBufferLength = (DWORD)(buffer.size() * sizeof(int16_t));
        header.dwFlags = 0;

        if (waveOutPrepareHeader(m_hWaveOut, &header, sizeof(WAVEHDR)) == MMSYSERR_NOERROR) {
            waveOutWrite(m_hWaveOut, &header, sizeof(WAVEHDR));
            
            // Wait for buffer to finish playing
            while (!(header.dwFlags & WHDR_DONE)) {
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
            }
            
            waveOutUnprepareHeader(m_hWaveOut, &header, sizeof(WAVEHDR));
        }
    }

    std::queue<std::pair<int, int>> m_queue;
    std::mutex m_mtx;
    std::condition_variable_any m_cv;
    std::jthread m_worker;
    HWAVEOUT m_hWaveOut = nullptr;
    std::atomic<bool> m_muted{false};
};

}  // namespace arc

#endif  // _MSC_VER

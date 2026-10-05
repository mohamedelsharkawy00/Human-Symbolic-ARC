// EVERY PUZZLE AT ONCE, ONE PER CPU THREAD, and what each came to handed back
// in the order the puzzles were asked. The reading shares nothing between
// puzzles, so any thread can read any one; the order the results come out in
// stays the order asked, so a run compares line for line with the last.
//   ARC_THREADS=n   reads on n threads, 1 for one puzzle at a time; unset, on
//                   every thread the CPU has.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <thread>
#include <vector>

// How many threads read at once: ARC_THREADS, or every thread the CPU has.
inline int reading_threads() {
    const char* asked = std::getenv("ARC_THREADS");
    const int threads = asked ? std::atoi(asked) : 0;
    return threads > 0 ? threads : std::max(1, (int)std::thread::hardware_concurrency());
}

// read(i) for every i below `count`, each on whichever thread is free; then, on
// this thread, in_order(i) for 0, 1, 2 ... each as soon as it and every one
// before it has been read.
template <class Read, class InOrder>
void in_parallel(int count, Read read, InOrder in_order) {
    std::vector<char> done((size_t)std::max(count, 0), 0);
    std::mutex guard;
    std::condition_variable finished;
    std::atomic<int> next{ 0 };
    std::vector<std::thread> threads;
    for (int t = std::min(reading_threads(), count); t > 0; --t)
        threads.emplace_back([&] {
            for (int i = next++; i < count; i = next++) {
                read(i);
                { const std::lock_guard lock(guard); done[(size_t)i] = 1; }
                finished.notify_one();
            }
        });
    for (int i = 0; i < count; ++i) {
        std::unique_lock lock(guard);
        finished.wait(lock, [&] { return done[(size_t)i] != 0; });
        lock.unlock();
        in_order(i);
    }
    for (std::thread& thread : threads) thread.join();
}

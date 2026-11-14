// How fast can this machine move bytes from memory to the CPU?
//
// Sums the same large array three ways:
//   1. in order, with a plain loop          -> the machine's streaming speed in GB/s
//   2. in order, following a list of indices
//   3. in random order, following the same list after shuffling it
//
// Walks 2 and 3 run the exact same code and do the exact same additions.
// Only the order of the visits differs, so any gap between them is the
// cost of fetching memory out of order.
//
// Build and run:
//     clang++ -O2 -std=c++17 memory_bandwidth.cpp -o memory_bandwidth
//     ./memory_bandwidth

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

using Clock = std::chrono::steady_clock;

const size_t N = 64'000'000;   // 64 million numbers x 8 bytes = 0.51 GB, far bigger than any cache
const int REPEATS = 3;         // time each walk 3 times, keep the fastest

// A benchmarking trap: an optimizing compiler only cares that the final answer
// is right, not when the work happens. Left alone it will reuse an old answer,
// or do the adding after the stopwatch has stopped, and every walk appears to
// take 0 seconds. This empty instruction says "the memory at p may be read or
// changed right here", which forces the real work to happen where we time it.
void no_shortcuts(const void* p) { asm volatile("" : : "g"(p) : "memory"); }

// Walk 1: visit a[0], a[1], a[2], ... directly.
uint64_t sum_plain(const std::vector<uint64_t>& a) {
    no_shortcuts(a.data());
    uint64_t total = 0;
    for (size_t i = 0; i < a.size(); i++) total += a[i];
    return total;
}

// Walks 2 and 3: visit a[] in whatever order the index list says.
uint64_t sum_by_index(const std::vector<uint64_t>& a, const std::vector<uint32_t>& order) {
    no_shortcuts(a.data());
    no_shortcuts(order.data());
    uint64_t total = 0;
    for (size_t i = 0; i < order.size(); i++) total += a[order[i]];
    return total;
}

// Runs one walk REPEATS times and returns the fastest time in seconds.
template <typename Walk>
double fastest_seconds(Walk walk, uint64_t& total) {
    double best = 1e9;
    for (int r = 0; r < REPEATS; r++) {
        auto start = Clock::now();
        total = walk();
        no_shortcuts(&total);                         // the sum must exist before the clock stops
        std::chrono::duration<double> elapsed = Clock::now() - start;
        best = std::min(best, elapsed.count());
    }
    return best;
}

void report(const char* name, double seconds, uint64_t total) {
    double gb = N * sizeof(uint64_t) / 1e9;
    std::printf("%-30s %8.3f %8.1f %15.2f   (sum %llu)\n", name, seconds, gb / seconds,
                seconds / N * 1e9, (unsigned long long)total);
}

int main() {
    std::vector<uint64_t> a(N);
    std::iota(a.begin(), a.end(), 0);                 // a = 0, 1, 2, ...

    std::vector<uint32_t> order(N);
    std::iota(order.begin(), order.end(), 0);         // order = 0, 1, 2, ... (in order)

    std::printf("array: %zu numbers x 8 bytes = %.2f GB\n\n", N, N * sizeof(uint64_t) / 1e9);
    std::printf("%-30s %8s %8s %15s\n", "walk", "seconds", "GB/s", "ns per number");

    uint64_t total = 0;
    double t_plain = fastest_seconds([&] { return sum_plain(a); }, total);
    report("in order, plain loop", t_plain, total);

    double t_ordered = fastest_seconds([&] { return sum_by_index(a, order); }, total);
    report("in order, via index list", t_ordered, total);

    std::mt19937_64 rng(42);
    std::shuffle(order.begin(), order.end(), rng);    // same indices, random order

    double t_random = fastest_seconds([&] { return sum_by_index(a, order); }, total);
    report("random order, via index list", t_random, total);

    std::printf("\nrandom order is %.0fx slower than in order (same code, same additions)\n",
                t_random / t_ordered);
    return 0;
}

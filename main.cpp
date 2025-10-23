// main.cpp
// High-performance maze solver (C++20) with OpenMP preprocessing.
// Build: g++ -O3 -march=native -fopenmp -std=c++20 main.cpp -o maze
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <limits>
#include <chrono>
#include <atomic>
#include <iostream>

using u8 = uint8_t;
using i32 = int32_t;
using u32 = uint32_t;
using usize = size_t;

static inline bool is_red_pixel(const u8* img, usize byteIdx) {
    u8 r = img[byteIdx], g = img[byteIdx+1], b = img[byteIdx+2];
    return r > 200 && g < 80 && b < 80;
}
static inline bool is_blue_pixel(const u8* img, usize byteIdx) {
    u8 r = img[byteIdx], g = img[byteIdx+1], b = img[byteIdx+2];
    return b > 200 && r < 80 && g < 80;
}
static inline bool is_walkable_pixel(const u8* img, usize byteIdx) {
    // near-white threshold
    int brightness = (int)img[byteIdx] + (int)img[byteIdx+1] + (int)img[byteIdx+2];
    return brightness >= 600;
}

static void hsv_to_rgb(float h, float s, float v, u8 &out_r, u8 &out_g, u8 &out_b) {
    float c = v * s;
    float hh = h / 60.0f;
    float x = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
    float r1=0, g1=0, b1=0;
    if (hh < 1)      { r1 = c; g1 = x; b1 = 0; }
    else if (hh < 2) { r1 = x; g1 = c; b1 = 0; }
    else if (hh < 3) { r1 = 0; g1 = c; b1 = x; }
    else if (hh < 4) { r1 = 0; g1 = x; b1 = c; }
    else if (hh < 5) { r1 = x; g1 = 0; b1 = c; }
    else             { r1 = c; g1 = 0; b1 = x; }
    float m = v - c;
    out_r = (u8)std::lround((r1 + m) * 255.0f);
    out_g = (u8)std::lround((g1 + m) * 255.0f);
    out_b = (u8)std::lround((b1 + m) * 255.0f);
}

int main(int argc, char** argv) {
    const std::string infile = (argc > 1) ? argv[1] : "bruh.png";
    const std::string outfile = (argc > 2) ? argv[2] : "solved.png";

    int width=0, height=0, channels=0;
    u8* image = stbi_load(infile.c_str(), &width, &height, &channels, 3);
    if (!image) {
        std::fprintf(stderr, "Failed to load '%s'\n", infile.c_str());
        return 1;
    }
    std::fprintf(stderr, "Loaded '%s' (%d x %d)\n", infile.c_str(), width, height);

    const usize W = (usize)width;
    const usize H = (usize)height;
    const usize total_pixels = W * H;
    if (total_pixels == 0) {
        std::fprintf(stderr, "Zero-size image\n");
        stbi_image_free(image);
        return 1;
    }

    // Timers (optional)
    auto tstart = std::chrono::steady_clock::now();

    // 1) Precompute walkable mask and find start/goal in parallel (OpenMP)
    std::vector<u8> walkable;
    walkable.assign(total_pixels, 0);

    std::atomic<int> start_idx(-1);
    std::atomic<int> goal_idx(-1);

    // Parallel scan: compute walkable mask and first red/blue pixels.
    // Using OpenMP to utilize multiple cores for large images.
    #pragma omp parallel for schedule(static)
    for (long long pi = 0; pi < (long long)total_pixels; ++pi) {
        usize i = (usize)pi;
        usize byteIdx = i * 3;
        bool w = is_walkable_pixel(image, byteIdx);
        walkable[i] = (u8)w;

        if (start_idx.load(std::memory_order_relaxed) < 0) {
            if (is_red_pixel(image, byteIdx)) {
                int expected = -1;
                start_idx.compare_exchange_strong(expected, (int)i, std::memory_order_relaxed);
            }
        }
        if (goal_idx.load(std::memory_order_relaxed) < 0) {
            if (is_blue_pixel(image, byteIdx)) {
                int expected = -1;
                goal_idx.compare_exchange_strong(expected, (int)i, std::memory_order_relaxed);
            }
        }
    }

    int s_idx = start_idx.load();
    int g_idx = goal_idx.load();
    if (s_idx < 0 || g_idx < 0) {
        std::fprintf(stderr, "Could not find red start or blue goal. start=%d goal=%d\n", s_idx, g_idx);
        stbi_image_free(image);
        return 1;
    }

    // 2) BFS (single-threaded) using a fast ring-buffer queue
    std::vector<i32> parent;
    try {
        parent.assign(total_pixels, -1);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "Failed to allocate parent array\n");
        stbi_image_free(image);
        return 1;
    }

    // ring buffer queue:
    std::vector<u32> q;
    try {
        q.resize(total_pixels); // allocate once
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "Failed to allocate queue\n");
        stbi_image_free(image);
        return 1;
    }

    usize head = 0, tail = 0;
    auto push = [&](u32 v) { q[tail++] = v; };
    auto pop = [&]() { return q[head++]; };
    auto empty = [&]() { return head == tail; };

    const u32 start_u = (u32)s_idx;
    const u32 goal_u  = (u32)g_idx;

    push(start_u);
    parent[start_u] = -2; // start marker

    const u32 Wu = (u32)W;
    bool found = false;

    // 4-neighbor offsets
    const int dx[4] = {1,-1,0,0};
    const int dy[4] = {0,0,1,-1};

    // BFS loop
    while (!empty()) {
        u32 cur = pop();
        if (cur == goal_u) { found = true; break; }

        u32 cx = cur % Wu;
        u32 cy = cur / Wu;
        // unroll neighbors
        for (int k = 0; k < 4; ++k) {
            int nx = (int)cx + dx[k];
            int ny = (int)cy + dy[k];
            if ((unsigned)nx >= W || (unsigned)ny >= H) continue;
            usize nidx = (usize)ny * W + (usize)nx;

            // visited check
            if (parent[nidx] != -1) continue;

            // walkable or goal
            if (!walkable[nidx] && (u32)nidx != goal_u) continue;

            parent[nidx] = (i32)cur;
            push((u32)nidx);
        }
    }

    if (!found) {
        std::fprintf(stderr, "No path found between start and goal.\n");
        stbi_image_free(image);
        return 1;
    }

    // Reconstruct path (vector) and color it
    // first find path length by walking parents
    std::vector<u32> path;
    path.reserve(1024);
    i32 cur = (i32)goal_u;
    while (cur != -2) {
        path.push_back((u32)cur);
        cur = parent[cur];
        if (cur == -1) {
            std::fprintf(stderr, "Parent chain broken\n");
            stbi_image_free(image);
            return 1;
        }
    }
    // path currently goal->start, reverse in-place
    std::reverse(path.begin(), path.end());
    const usize path_len = path.size();

    for (usize i = 0; i < path_len; ++i) {
        float t = (path_len <= 1) ? 0.0f : (float)i / (float)(path_len - 1);
        float hue = t * 360.0f;
        u8 r,g,b;
        hsv_to_rgb(hue, 1.0f, 1.0f, r, g, b);
        usize idx = (usize)path[i];
        usize byteIdx = idx * 3;
        image[byteIdx + 0] = r;
        image[byteIdx + 1] = g;
        image[byteIdx + 2] = b;
    }

    if (!stbi_write_png(outfile.c_str(), width, height, 3, image, width * 3)) {
        std::fprintf(stderr, "Failed to write output '%s'\n", outfile.c_str());
        stbi_image_free(image);
        return 1;
    }

    auto tend = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(tend - tstart).count();
    std::fprintf(stderr, "Solved. Path length=%zu, time=%.3fs, output='%s'\n", path_len, sec, outfile.c_str());

    stbi_image_free(image);
    return 0;
}

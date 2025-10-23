// main.cpp
// C++20 friendly maze solver that detects RED start and BLUE goal,
// treats near-white as walkable, BFS to find shortest path,
// paints the path with a full-spectrum hue gradient, and writes solved.png.
//
// Put stb_image.h and stb_image_write.h in the same folder.
// Build with your CLion/CMake setup (C++20).

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <iostream>
#include <string>
#include <limits>
#include <unistd.h> // For getcwd
#include <deque>

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
    // Accept near-white (brightness threshold)
    int brightness = (int)img[byteIdx] + (int)img[byteIdx+1] + (int)img[byteIdx+2];
    // brightness range 0..765 → threshold ~ 600 corresponds to ~200 average
    return brightness >= 600;
}

static void hsv_to_rgb(float h, float s, float v, u8 &out_r, u8 &out_g, u8 &out_b) {
    // h in [0,360)
    float c = v * s;
    float hh = h / 60.0f;
    float x = c * (1.0f - fabsf(fmodf(hh, 2.0f) - 1.0f));
    float r1=0, g1=0, b1=0;
    if (hh < 0) hh = 0;
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

    // quick working dir print to debug CLion working dir issues
    {
        char cwd[4096];
        if (getcwd(cwd, sizeof(cwd))) {
            std::fprintf(stderr, "Working directory: %s\n", cwd);
        }
    }

    int width=0, height=0, channels=0;
    // load 3-channel RGB
    u8* image = stbi_load(infile.c_str(), &width, &height, &channels, 3);
    if (!image) {
        std::fprintf(stderr, "Failed to load '%s'\n", infile.c_str());
        return 1;
    }
    std::fprintf(stderr, "Loaded '%s' (%d x %d), channels = %d\n", infile.c_str(), width, height, 3);

    const usize W = (usize)width;
    const usize H = (usize)height;
    const usize total_pixels = W * H;
    if (total_pixels == 0) {
        std::fprintf(stderr, "Zero-size image\n");
        stbi_image_free(image);
        return 1;
    }

    // Pre-calculate walkable areas to speed up BFS
    std::vector<bool> walkable(total_pixels);
    for (usize i = 0; i < total_pixels; ++i) {
        walkable[i] = is_walkable_pixel(image, i * 3);
    }

    // find red start and blue goal (pick first occurrence)
    i32 start_idx = -1;
    i32 goal_idx  = -1;
    for (usize pi = 0; pi < total_pixels; ++pi) {
        const usize byteIdx = pi * 3;
        if (start_idx < 0 && is_red_pixel(image, byteIdx)) {
            start_idx = (i32)pi;
        }
        if (goal_idx < 0 && is_blue_pixel(image, byteIdx)) {
            goal_idx = (i32)pi;
        }
        if (start_idx >= 0 && goal_idx >= 0) break;
    }

    if (start_idx < 0 || goal_idx < 0) {
        std::fprintf(stderr, "Could not find red start or blue goal. start=%d goal=%d\n", start_idx, goal_idx);
        stbi_image_free(image);
        return 1;
    }

    std::fprintf(stderr, "Start pixel index = %d, goal index = %d\n", start_idx, goal_idx);

    // BFS structures
    // parent: store parent pixel index (int32_t), -1 means none.
    // This also implicitly functions as the visited set.
    std::vector<i32> parent;
    try {
        parent.assign(total_pixels, -1);
    } catch (const std::bad_alloc&) {
        std::fprintf(stderr, "Failed to allocate parent array (out of memory)\n");
        stbi_image_free(image);
        return 1;
    }

    std::deque<u32> queue;

    const u32 start_u = (u32)start_idx;
    const u32 goal_u  = (u32)goal_idx;

    queue.push_back(start_u);
    parent[start_u] = -2; // Use -2 to mark start as visited with no parent

    std::fprintf(stderr, "Starting BFS...\n");

    const int dx[4] = {1,-1,0,0};
    const int dy[4] = {0,0,1,-1};
    bool found = false;
    u32 steps = 0;

    while (!queue.empty()) {
        u32 cur = queue.front();
        queue.pop_front();

        if (cur == goal_u) { found = true; break; }
        u32 cx = cur % (u32)W;
        u32 cy = cur / (u32)W;
        for (int k = 0; k < 4; ++k) {
            int nx = (int)cx + dx[k];
            int ny = (int)cy + dy[k];
            if (nx < 0 || ny < 0 || (usize)nx >= W || (usize)ny >= H) continue;
            usize nidx = (usize)ny * W + (usize)nx;
            if (parent[nidx] != -1) continue; // Already visited

            bool is_goal = (nidx == goal_u);
            if (!walkable[nidx] && !is_goal) continue;

            parent[nidx] = (i32)cur;
            queue.push_back((u32)nidx);
        }

        if (++steps % 10000000 == 0) {
            std::fprintf(stderr, "BFS progress: queue.size=%zu\n", queue.size());
        }
    }

    if (!found) {
        std::fprintf(stderr, "No path found between start and goal.\n");
        stbi_image_free(image);
        return 1;
    }

    std::fprintf(stderr, "Path found. Reconstructing path...\n");

    // reconstruct path indices (deque holds indices from start -> goal)
    std::deque<u32> path;
    i32 cur = (i32)goal_u;
    while (cur != -2) { // -2 is the start marker
        path.push_front((u32)cur);
        cur = parent[cur];
        if (cur == -1 || path.size() > total_pixels) {
            std::fprintf(stderr, "Parent chain corrupt or broken, aborting\n");
            stbi_image_free(image);
            return 1;
        }
    }

    const usize path_len = path.size();
    std::fprintf(stderr, "Path length = %zu\n", path_len);

    // paint the path with full hue spectrum
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

    // write output
    if (!stbi_write_png(outfile.c_str(), width, height, 3, image, width * 3)) {
        std::fprintf(stderr, "Failed to write output '%s'\n", outfile.c_str());
        stbi_image_free(image);
        return 1;
    }

    std::fprintf(stderr, "Saved solved image to '%s'\n", outfile.c_str());

    stbi_image_free(image);
    return 0;
}

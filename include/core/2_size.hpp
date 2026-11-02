// This file determines what decides the output's width and height.
//
// The first question the reading order asks, and the one that needs the least:
// infer_rule compares the width and height of each train pair and nothing else,
// so it never opens a grid. That is why it sits below 1_readings.hpp rather
// than beside it — reading a grid is not one of its needs.
//
// Its rule gates the pair-relation detectors (prepare_figures, 13_infer.hpp):
// only an Identity output overlays its input cell for cell, so only then are
// they meaningful.

#pragma once
#include "0_core.hpp"
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace arc {
using namespace std;

// What decides the output size: read off the train pairs by infer_rule(),
// then read by the replays and the ops.
struct output_dimension_rule {
    enum : int8_t {
        NotInferredYet,
        Identity,
        Fixed,
        Scale,
        BoundingBox
    } Chosen = NotInferredYet;

    // What the chosen case learned, 0 = that case never uses this number
    int width = 0, height = 0;
    int width_numerator = 0, width_denominator = 0;
    int height_numerator = 0, height_denominator = 0;
    // WHERE THE INPUT LIES ON THE CANVAS: one of CopyTo's nine anchors
    // (anchor_names, aligned_copy_start), 0 the top-left, where the input lies
    // unless the pairs read it elsewhere (input_laid_on).
    int8_t anchor = 0;

    // True once a sizing policy is selected, including the BoundingBox fallback.
    bool holds() const { return Chosen != NotInferredYet; }

    // The canvas this rule gives one input before anything is painted on it.
    // False under BoundingBox, where only the drawing says, and when the size
    // would not fit a grid at all.
    bool size_for(int input_width, int input_height,
                  int& out_width, int& out_height) const {
        out_width = out_height = 0;
        long long w = 0, h = 0;
        switch (Chosen) {
        case Identity: w = input_width; h = input_height; break;
        case Fixed: w = width; h = height; break;
        case Scale:
            if (width_denominator <= 0 || height_denominator <= 0 ||
                width_numerator <= 0 || height_numerator <= 0) return false;
            w = (long long)input_width * width_numerator / width_denominator;
            h = (long long)input_height * height_numerator / height_denominator;
            break;
        case BoundingBox:
        case NotInferredYet:
        default: return false;
        }
        if (w <= 0 || h <= 0 || w > MAX_GRID_DIM || h > MAX_GRID_DIM) return false;
        out_width = (int)w; out_height = (int)h;
        return true;
    }

    // The working frame is shared by the replays and the ops. BoundingBox
    // alone gets the full workspace; its final bounds come from painted cells.
    // Invalid or unknown rules give no frame, never an implicit Identity.
    bool canvas_size_for(int input_width, int input_height,
                         int& out_width, int& out_height) const {
        if (Chosen == BoundingBox) {
            out_width = out_height = MAX_GRID_DIM;
            return true;
        }
        return size_for(input_width, input_height, out_width, out_height);
    }

    // Picks how output size is decided, first match wins
    static output_dimension_rule infer_rule(const vector<example>& train) {
        if (train.empty())
            return { .Chosen = BoundingBox };

        // Lowest terms (3/7 not 9/21), so a no-change ratio is always exactly
        // 1/1 — the Identity-vs-Scale check below depends on that
        auto reduce = [](int numerator, int denominator) -> std::pair<int, int> {
            if (numerator <= 0 || denominator <= 0)
                return { 0, 0 };
            const int shared = std::gcd(numerator, denominator);
            return { numerator / shared, denominator / shared };
        };

        // Separate ratios: a transposed 12x16 -> 16x12 has equal area, so one
        // combined number would wrongly call it "same size"
        auto [width_numerator, width_denominator] = reduce(train[0].output.width,  train[0].input.width);
        auto [height_numerator, height_denominator] = reduce(train[0].output.height, train[0].input.height);

        const bool has_ratio = width_denominator != 0 && height_denominator != 0 &&
                               ranges::all_of(train, [&](const example& ex) {
            // Cross-multiplied: division would truncate and false-confirm the
            // ratio on non-ratio puzzles (e.g. 7->3 then 10->4 "passes" 3/7)
            return !ex.input.empty() &&
                   ex.output.width  * width_denominator == ex.input.width  * width_numerator &&
                   ex.output.height * height_denominator == ex.input.height * height_numerator;
        });

        // Every pair keeps its size
        if (has_ratio && width_numerator == width_denominator && height_numerator == height_denominator)
            return { .Chosen = Identity };

        // Every pair scales by the same fraction
        if (has_ratio)
            return { .Chosen = Scale,
                     .width_numerator = width_numerator, .width_denominator = width_denominator,
                     .height_numerator = height_numerator, .height_denominator = height_denominator };

        // Every output lands on one size, whatever the input was
        const int width = train[0].output.width, height = train[0].output.height;
        if (ranges::all_of(train, [&](const example& ex) {
                return ex.output.width == width && ex.output.height == height; }))
            return { .Chosen = Fixed, .width = width, .height = height };

        // Nothing held, so the size comes from what the program draws
        return { .Chosen = BoundingBox };
    }
};

// Do input and output overlay cell-for-cell in EVERY pair? The detectors in
// 7_facts.hpp and 9_rules.hpp that read out.data at an input coordinate need this
// to be true, and each one says so itself rather than trusting the caller —
// infer() gates on the size rule, but a detector called from anywhere else must
// still be safe.
inline bool pairs_overlay(const vector<example>& train) {
    if (train.empty()) return false;
    return ranges::all_of(train, [](const example& ex) {
        // Non-empty as well as equal: an empty pair overlays itself vacuously,
        // and a channel that divides by the cell count would hand back a
        // nonsense number instead of declining.
        return !ex.input.empty() && ex.input.same_size(ex.output);
    });
}

}  // namespace arc

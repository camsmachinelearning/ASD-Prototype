//
//  xoshiro256.h
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/17/25.
//

#pragma once
#include <random>
#include <cstdint>


class Xoshiro256 {
private:
    uint64_t state[4];

    static inline uint64_t rotl(const uint64_t x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    explicit Xoshiro256(uint64_t seed = std::random_device()()) {
        seed_engine(seed);
    }

    void seed_engine(uint64_t seed = std::random_device()()) {
        std::mt19937_64 seeder(seed);
        for (uint64_t & i : state)
            i = seeder();
    }

    uint64_t next() {
        const uint64_t result = rotl(state[0] + state[3], 23) + state[0];

        const uint64_t t = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= t;
        state[3] = rotl(state[3], 45);

        return result;
    }

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }
};


#pragma once
#include <random>
#include <cstdint>

class Xoshiro128 {
private:
    uint32_t state[4];

    static inline uint32_t rotl(const uint32_t x, int k) {
        return (x << k) | (x >> (32 - k));
    }

public:
    explicit Xoshiro128(uint32 seed = std::random_device()()) {
        seed_engine(seed);
    }
    
    void seed_engine(uint32_t seed = std::random_device()()) {
        std::mt19937 seeder(seed);
        for (auto & i : state)
            i = seeder();
    }
    
    
    uint32_t next() {
        const uint32_t result = rotl(state[0] + state[3], 7) + state[0];
        const uint32_t t = state[1] << 9;
       
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];
        
        state[2] ^= t;
        state[3] = rotl(state[3], 11);
        
        return result;
    }
        
    static constexpr uint32_t min() { return 0; }
    static constexpr uint32_t max() { return UINT32_MAX; }
};


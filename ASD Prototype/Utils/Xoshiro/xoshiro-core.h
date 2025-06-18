#pragma once

class Xoshiro128 {
private:
    using uint32 = uint32
    uint32 state[4];

    static inline uint32 rotl(const uint32 x, int k) {
        return (x << k) | (x >> (32 - k));
    }

public:
    Xoshiro256(uint32 seed = std::random_device()()) {
        seed_engine(seed);
    }

    void seed_engine(uint32_t seed = std::random_device()()) {
        std::mt19937 seeder(seed);
        for (auto & i : state)
            i = seeder();
    }

    uint32 operator()() {
        const uint32 result = rotl(state[0] + state[3], 7) + state[0];

        const uint32 t = state[1] << 9;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= t;
        state[3] = rotl(state[3], 11);

        return result;
    }

    static constexpr uint32 min() { return 0; }
    static constexpr uint32 max() { return UINT32_MAX; }
};


class Xoshiro256 {
private:
    using uint64 = unsigned long long;
    uint64 state[4];

    static inline uint64 rotl(const uint64 x, int k) {
        return (x << k) | (x >> (64 - k));
    }

public:
    explicit Xoshiro256(uint64 seed = std::random_device()()) {
        seed_engine(seed);
    }

    void seed_engine(uint64 seed = std::random_device()()) {
        std::mt19937_64 seeder(seed);
        for (uint64 & i : state)
            i = seeder();
    }

    uint32_t operator()() {
        const uint64 result = rotl(state[0] + state[3], 23) + state[0];

        const uint64 t = state[1] << 17;
        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= t;
        state[3] = rotl(state[3], 45);

        return result;
    }

    static constexpr uint64 min() { return 0; }
    static constexpr uint64 max() { return UINT64_MAX; }
};

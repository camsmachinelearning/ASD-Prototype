//
//  SplitMix64.h
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/18/25.
//

#ifndef SplitMix64_h
#define SplitMix64_h

#include <random>
#include <cstdint>

class SplitMix64 {
private:
    uint64_t x;

public:
    SplitMix64() : x(std::random_device()()) {}
    SplitMix64(uint64_t seed) : x(seed) {}
    
    void seed(uint64_t seed) {
        this->x = seed;
    }
    
    void seed() {
        this->x = std::random_device()();
    }

    uint64_t next() {
        uint64_t z = (x += 0x9E3779B97f4A7C15);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
        return z ^ (z >> 31);
    }
};

#endif /* SplitMix64_h */

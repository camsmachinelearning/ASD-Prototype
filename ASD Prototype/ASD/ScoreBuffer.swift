//
//  ScoreBuffer.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/29/25.
//

import Foundation
import CoreML


extension ASD {
    final class ScoreBuffer: Buffer {
        typealias Element = Score
        
        internal struct Score {
            var cumulativeScore: Float = 0
            var hits: UInt32 = 0
            
            var score: Float {
                return cumulativeScore//self.hits > 0 ? self.cumulativeScore / Float(self.hits) : 0
            }
            
            mutating func update(with score: Float) {
                self.cumulativeScore = score
                self.hits += 1
            }
        }
        
        // MARK: Attributes
        
        var count: Int { self.bufferSize }
        
        private var buffer: ContiguousArray<Score>
        private var writeIndex: Int
        private var bufferSize: Int { self.buffer.count }
        
        // MARK: Constructors
        
        public init(atTime time: Double, frontPadding: Int = 25, capacity: Int = 53) {
            self.writeIndex = frontPadding
            self.buffer = .init(repeating: .init(), count: capacity)
        }
        
        // MARK: Subscripting
        
        public subscript (_ index: Int) -> Float {
            get {
                return self.buffer[self.wrapIndex(index)].score
            }
            set {
                return self.buffer[self.wrapIndex(index)].update(with: newValue)
            }
        }
        
        // MARK: Buffer
        
        func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Score>) throws -> R) rethrows -> R {
            return try self.buffer.withUnsafeBufferPointer(body)
        }
        
        func withUnsafeMutableBufferPointer<R>(_ body: (inout UnsafeMutableBufferPointer<Score>) throws -> R) rethrows -> R {
            return try self.buffer.withUnsafeMutableBufferPointer(body)
        }
        
        // MARK: public methods
        
        /// Write to the buffer
        /// - Parameters:
        ///   - time the time at which the input was processed or added
        ///   - source the data source from which to write
        ///   - offset how many indices to skip
        public func write(from source: MLMultiArray, offsetBy offset: Int = 0) {
            var i = Utils.mod(self.writeIndex + offset - source.count, self.bufferSize)
            
            source.withUnsafeBufferPointer(ofType: Float.self) { ptr in
                for score in ptr {
                    self.buffer[i].update(with: score)
                    i += 1
                    if i == self.bufferSize {
                        i = 0
                    }
                }
                self.writeIndex = i
            }
        }
        
        public func read(at index: Int) -> Float {
            return self.buffer[self.wrapIndex(index)].score
        }
        
        // MARK: private helpers
        @inline(__always)
        private func wrapIndex(_ index: Int) -> Int {
            return Utils.wrapIndex(index, start: self.writeIndex - self.bufferSize, end: self.writeIndex, capacity: self.bufferSize)
        }
    }
}

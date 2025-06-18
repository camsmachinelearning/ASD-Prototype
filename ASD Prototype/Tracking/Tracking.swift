//
//  Face.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/16/25.
//

import Foundation
import CoreML
import Accelerate
import Xoshiro


struct Detection {
    let box: CGRect
    let embedding: MLMultiArray
    let confidence: Float
}

class Tracker {
    public private(set) var activeTracks: IndexedSet<Track>
    public private(set) var inactiveTracks: Set<Track>
    
    public var lambda: Float = 0.98
    public var maxCosineDistance: Float = 0.3
    public var maxMahanobisDistance: Float = 9.4877
    
    public let maxCost: Float = 1e30
    
    private var idGenerator: Xoshiro256
    private var currentTime: UInt = 0
    private let maxDisappearanceTime: UInt = 25
    
    init() {
        self.activeTracks = IndexedSet<Track>()
        self.inactiveTracks = Set<Track>()
    }
    
    func update(detections: [Detection]) {
        
        var costMatrix: [Float] = []
        costMatrix.reserveCapacity(activeTracks.count * detections.count)

        for track in activeTracks {
            // update predictions
            track.kalmanFilter.predict()
            track.update()
            
            // match detection
            for detection in detections {
                let box = detection.box
                let features = detection.embedding
                
                let motionCost: Float = track.kalmanFilter.computeMotionCost(measured: box)
                let appearanceCost: Float = Tracker.cosineDistance(a: features, b: track.features)
                costMatrix.append(self.combineCosts(motionCost: motionCost, appearanceCost: appearanceCost))
            }
        }
        
        // update matching tracks
        var rows: [Int32] = []
        var cols: [Int32] = []
        var unusedDetections: [Bool] = [Bool](repeating: true, count: detections.count)
        var unusedTracks: [Bool] = [Bool](repeating: true, count: activeTracks.count)
        
        solveLap(inputDims: activeTracks.count, outputDims: detections.count, cost: costMatrix, rowSolution: &rows, colSolution: &cols)
        
        for (row32, col32) in zip(rows, cols) {
            let row = Int(row32)
            let col = Int(col32)
            let index = row * detections.count + col
            
            if costMatrix[index] >= self.maxCost {
                continue
            }
            
            unusedTracks[row] = false
            unusedDetections[col] = false
            
            activeTracks[row].kalmanFilter.update(measurement: detections[col].box)
            activeTracks[row].registerHit(time: currentTime)
        }
        
        // update unmatched tracks
        var deactivatedIDs: Set<UInt> = []
        var deletedIDs: Set<UInt> = []
        
        for (i, track) in activeTracks.enumerated() where unusedTracks[i] {
            if track.canDelete {
                deletedIDs.insert(track.id)
            } else if self.currentTime - track.timeWhenLastDetected > self.maxDisappearanceTime {
                deactivatedIDs.insert(track.id)
                track.deactivate()
            }
        }
        
        for id in deactivatedIDs {
            self.inactiveTracks.insert(activeTracks.remove(id: id)!)
        }
        for id in deletedIDs {
            activeTracks.remove(id: id)
        }
        
        // handle new detections
        for (i, detection) in detections.enumerated() where unusedDetections[i] {
            let track = Track(
                id: UInt(self.idGenerator.next()),
                box: detection.box,
                features: detection.embedding,
                time: self.currentTime
            )
            
            activeTracks.append(track)
        }
        
        self.currentTime += 1
    }
    
    private static func intersectionOverUnion(_ box1: CGRect, _ box2: CGRect) -> Float {
        let intersectionWidth = min(box1.maxX, box2.maxX) - max(box1.minX, box2.minX)
        let intersectionHeight = min(box1.maxY, box2.maxY) - max(box1.minY, box2.minY)
        
        if intersectionWidth <= 0 || intersectionHeight <= 0 {
            return 0
        }
        
        let area1 = box1.width * box1.height
        let area2 = box2.width * box2.height
        let areaIntersection = intersectionWidth * intersectionHeight
        let areaUnion = area1 + area2 - areaIntersection
        
        return Float(areaIntersection) / Float(areaUnion)
    }
    
    private static func makeMeasurementVector(rect: CGRect) -> SIMD4<Float> {
        return SIMD4<Float>(
            Float(rect.midX),
            Float(rect.midY),
            Float(rect.width * rect.height),
            Float(rect.width / rect.height)
        )
    }
    
    private static func cosineDistance(a: MLMultiArray, b: MLMultiArray) -> Float {
        // 1. Quick shape check
        guard a.dataType == .float32,
              b.dataType == .float32,
              a.count == 128,
              b.count == 128 else {
            return 2
        }

        // 2. Bind the raw pointers
        let aPtr = a.dataPointer.bindMemory(to: Float.self, capacity: 128)
        let bPtr = b.dataPointer.bindMemory(to: Float.self, capacity: 128)
        let n = vDSP_Length(128)

        // 3. Dot product
        var dot: Float = 0
        vDSP_dotpr(aPtr, 1, bPtr, 1, &dot, n)

        // 4. Squared norms
        var normASq: Float = 0
        var normBSq: Float = 0
        vDSP_svesq(aPtr, 1, &normASq, n)
        vDSP_svesq(bPtr, 1, &normBSq, n)
        
        let normANormB = sqrt(normASq * normBSq)

        // 6. Final cosine similarity
        return 1.0 - dot / normANormB
    }
    
    private func combineCosts(motionCost: Float, appearanceCost: Float) -> Float {
        if motionCost > maxMahanobisDistance || appearanceCost > maxMahanobisDistance {
            return maxCost + 1
        }
        return self.lambda * motionCost + (1 - self.lambda) * appearanceCost
    }
}

//
//  Track.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/17/25.
//

import CoreML
import Foundation


class Track:
    Identifiable,
    Hashable,
    Equatable
{
    let id: UInt
    let kalmanFilter: VisualKF
    var features: MLMultiArray
    
    var timeWhenLastDetected: UInt
    var hits: UInt
    var confirmed: Bool = false
    
    var canDelete: Bool {
        return !self.confirmed && self.hits < 0
    }
    
    private static let defaultConfirmationThreadhold: UInt = 3
    //private static let confirmationThreadhold: UInt = 3
    
    public init(id: UInt, box: CGRect, features: MLMultiArray, time: UInt) {
        self.id = id
        self.kalmanFilter = VisualKF(initialObservation: box)
        self.features = features
        self.timeWhenLastDetected = time
        self.hits = 0
    }
    
    static func == (lhs: Track, rhs: Track) -> Bool {
        return lhs.id == rhs.id // Compare properties
    }
    
    func update() {
        self.kalmanFilter.predict()
        self.hits -= 1
    }
    
    func registerHit(time: UInt, confirmationThreadhold: UInt = Track.defaultConfirmationThreadhold) {
        self.hits += 2
        self.timeWhenLastDetected = time
        
        if self.hits >= confirmationThreadhold {
            self.confirmed = true
        }
    }
    
    func deactivate() {
        self.kalmanFilter.xVelocity = 0
        self.kalmanFilter.yVelocity = 0
        self.kalmanFilter.growthRate = 0
        
        self.hits = 0
        
    }
    
    func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
}


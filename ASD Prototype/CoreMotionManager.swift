//
//  CoreMotionManager.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/12/25.
//

import Foundation
import CoreMotion

class CoreMotionManager {
    private let motionManager = CMMotionManager()

    var rotationRate: CMRotationRate? {
        return motionManager.deviceMotion?.rotationRate
    }

    init() {
        if motionManager.isDeviceMotionAvailable {
            motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
            motionManager.startDeviceMotionUpdates()
        }
    }

    deinit {
        motionManager.stopDeviceMotionUpdates()
    }
}

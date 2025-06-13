//
//  KalmanFilter.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/12/25.
//

import Foundation
import LANumerics

class KalmanFilter {
    // State vector (x, y, w, h, vx, vy, vw, vh)
    private(set) var x: Matrix<Float>

    // Covariance matrix
    private var P: Matrix<Float>

    // State transition matrix
    private let A: Matrix<Float>

    // Measurement matrix
    private let H: Matrix<Float>

    // Process noise covariance
    private var Q: Matrix<Float>

    // Measurement noise covariance
    private var R: Matrix<Float>
    
    // Identity matrix
    private let I: Matrix<Float>

    // Time step
    private let dt: Float = 1.0 / 30.0 // Assuming 30 FPS

    init(initialObservation: CGRect) {
        let stateSize = 8
        let defaultQNoise: Float = 1
        let defaultRNoise: Float = 0.1
        
        self.I = Matrix<Float>.eye(stateSize)
        
        self.x = Matrix<Float>([
            Float(initialObservation.midX),
            Float(initialObservation.midY),
            Float(initialObservation.width),
            Float(initialObservation.height),
            0,
            0,
            0,
            0
        ])
        
        self.P = 1000.0 * Matrix<Float>.eye(stateSize)

        self.A = Matrix(rows: [
            [1, 0, 0, 0, dt, 0, 0, 0],
            [0, 1, 0, 0, 0, dt, 0, 0],
            [0, 0, 1, 0, 0, 0, dt, 0],
            [0, 0, 0, 1, 0, 0, 0, dt],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.H = Matrix(rows: 4, columns: 8, diagonal: [Float](repeating: 1, count: 4))

        // These noise values might need tuning.
        self.Q = defaultQNoise * Matrix<Float>.eye(stateSize)
        self.R = defaultRNoise * Matrix<Float>.eye(4)
    }

    func predict() {
        x = A * x
        P = A * P * A.transpose + Q
    }

    func update(measurement: Vector<Float>) {
        let y = Matrix<Float>(measurement) - (H * x)
        let S = H * P * H.transpose + R
        guard let S_inv = S.inverse else { return }
        let K = P * H.transpose * S_inv
        x = x + (K * y)
        P = (I - K * H) * P
    }

    var predictedRect: CGRect {
        return CGRect(
            x: CGFloat(x[0] - x[2] / 2),
            y: CGFloat(x[1] - x[3] / 2),
            width: CGFloat(x[2]),
            height: CGFloat(x[3])
        )
    }
}

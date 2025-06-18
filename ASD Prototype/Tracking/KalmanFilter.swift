//
//  KalmanFilter.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/12/25.
//

import Foundation
import LANumerics
import simd

class VisualKF : KalmanFilter {
    // State: (x, y, s, r, vx, vy, s')
    // Measurement: (x, y, w, h)
    
    @inline(__always)
    public var x: Float {
        get {
            return state[0]
        }
        set {
            state[0] = newValue
        }
    }
    
    @inline(__always)
    public var y: Float {
        get {
            return state[1]
        }
        set {
            state[1] = newValue
        }
    }
    
    @inline(__always)
    public var scale: Float {
        get {
            return state[2]
        }
        set {
            state[2] = newValue
        }
    }
    
    @inline(__always)
    public var aspectRatio: Float {
        get {
            return state[3]
        }
        set {
            state[3] = newValue
        }
    }
    
    @inline(__always)
    public var xVelocity: Float {
        get {
            return state[4]
        }
        set {
            state[4] = newValue
        }
    }
    
    @inline(__always)
    public var yVelocity: Float {
        get {
            return state[5]
        }
        set {
            state[5] = newValue
        }
    }
    
    @inline(__always)
    public var growthRate: Float {
        get {
            return state[6]
        }
        set {
            state[6] = newValue
        }
    }
    
    public var width: Float {
        get {
            return sqrt(self.scale * self.aspectRatio)
        }
        set {
            self.scale = newValue * newValue / self.aspectRatio
        }
    }
    
    public var height: Float {
        get {
            return sqrt(self.scale / self.aspectRatio)
        }
        set {
            self.scale = newValue * newValue * self.aspectRatio
        }
    }
    
    public var rect: CGRect {
        let width = self.width
        let height = self.height
        return CGRect(
            x: CGFloat(x - width / 2),
            y: CGFloat(y - height / 2),
            width: CGFloat(width),
            height: CGFloat(height)
        )
    }
    
    private var dt: Float
    
    init(initialObservation: CGRect, dt: Float = 1.0 / 25.0) {
        self.dt = dt
        
        super.init(
            x: VisualKF.convertRectToMeasurement(initialObservation) + [0, 0, 0],
            A: Matrix(rows: [
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0]
            ]),
            B: Matrix.zeros(7, 7),
            H: Matrix(rows: 4, columns: 7, diagonal: [Float](repeating: 1, count: 4)),
            Q:   1 * Matrix<Float>.eye(8),
            R: 0.1 * Matrix<Float>.eye(4),
            dt: dt
        )
    }
    
    static func convertRectToMeasurement(_ rect: CGRect) -> [Float] {
        return [
            Float(rect.midX),
            Float(rect.midY),
            Float(rect.width * rect.height),
            Float(rect.width / rect.height),
        ]
    }
    
    override func predict() {
        let dtVec = SIMD3<Float>(repeating: self.dt)
        let dxVec = SIMD3<Float>(self.state.vector[4..<7]) * dtVec
        let xVec = SIMD3<Float>(self.state.vector[0..<3]) + dxVec
        
        self.state[0] = xVec.x
        self.state[1] = xVec.y
        self.state[2] = xVec.z
    }
    
    func computeMotionCost(measured: CGRect) -> Float {
        let measurement = SIMD4(
            x: Float(measured.midX),
            y: Float(measured.midY),
            z: Float(measured.width * measured.height),
            w: Float(measured.width / measured.height),
        )
        
        let y = measurement - (measurementMatrix * state).simd4
        let S = (measurementMatrix * covariance * measurementMatrix.transpose).simd4x4 + measurementNoiseCovariance.simd4x4
        let SInv = S.inverse
        return simd_dot(y, SInv * y)
    }
    
    func update(measurement: CGRect) {
        super.update(measurement: Vector<Float>(VisualKF.convertRectToMeasurement(measurement)))
    }
}

class KalmanFilter {
    // State vector
    public var state: Matrix<Float>

    // Covariance matrix
    public fileprivate(set) var covariance: Matrix<Float>

    // State transition matrix
    public let stateTransitionMatrix: Matrix<Float>
    
    // Control matrix
    public let controlMatrix: Matrix<Float>

    // Measurement matrix
    public let measurementMatrix: Matrix<Float>

    // Process noise covariance
    public var processNoiseCovariance: Matrix<Float>

    // Measurement noise covariance
    public var measurementNoiseCovariance: Matrix<Float>
    
    // Identity matrix
    public let I: Matrix<Float>

    init(x: Vector<Float>, A: Matrix<Float>, B: Matrix<Float>, H: Matrix<Float>, Q: Matrix<Float>, R: Matrix<Float>, dt: Float, covariance: Matrix<Float>? = nil) {
        self.I = Matrix<Float>.eye(x.count)
        
        self.state = Matrix<Float>(x)
        
        self.covariance = covariance ?? 1000.0 * Matrix<Float>.eye(x.count)

        self.stateTransitionMatrix = dt * A + Matrix<Float>.eye(x.count)
        self.controlMatrix = dt * B

        self.measurementMatrix = H

        // These noise values might need tuning.
        self.processNoiseCovariance = Q
        self.measurementNoiseCovariance = R
    }

    public func predict() {
        state = stateTransitionMatrix * state
        updateCovariancePredict()
    }
    
    public func predict(u: Vector<Float>) {
        state = stateTransitionMatrix * state + controlMatrix * Matrix<Float>(u)
        updateCovariancePredict()
    }
    
    @inline(__always)
    func updateCovariancePredict() {
        covariance = stateTransitionMatrix * covariance * stateTransitionMatrix.transpose + processNoiseCovariance
    }
    
    public func update(measurement: Vector<Float>) {
        let y = Matrix<Float>(measurement) - (measurementMatrix * state)
        let S = measurementMatrix * covariance * measurementMatrix.transpose + measurementNoiseCovariance
        guard let S_inv = S.inverse else { return }
        
        let K = covariance * measurementMatrix.transpose * S_inv
        state = state + (K * y)
        covariance = (I - K * measurementMatrix) * covariance
    }
    
    public func step(measurement: Vector<Float>) {
        predict()
        update(measurement: measurement)
    }
}

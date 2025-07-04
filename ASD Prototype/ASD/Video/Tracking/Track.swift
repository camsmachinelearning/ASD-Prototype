//
//  Track.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/17/25.
//

import CoreML
import Foundation

extension ASD.Tracking {
    final class Track:
        Identifiable,
        Hashable,
        Equatable
    {
        // MARK: public enums
        enum TrackInitializationError: Error {
            case missingEmbedding
        }
        
        enum Status {
            case active     /// track is actively in use
            case inactive   /// track is not actively in use but has been initialized
            case pending    /// track has not been confirmed
            
            var isActive: Bool { return self == .active }
            var isInactive: Bool { return self == .inactive }
            var isPending: Bool { return self == .pending }
            var isConfirmed: Bool { return self != .pending }
            
            var stringValue: String {
                switch self {
                case .active:
                    return "active"
                case .inactive:
                    return "inactive"
                case .pending:
                    return "pending"
                }
            }
        }
        
        // MARK: public properties
        
        public let id = UUID()
        
        public private(set) var hits: Int = 1
        public private(set) var costs: Costs = Costs()
        public private(set) var status: Status = .pending
        public private(set) var embedding: MLMultiArray
        public private(set) var averageAppearanceCost: Float
        
        // MARK: public computed properties
        public var isDeletable: Bool {
            return self.status.isPending && self.hits <= 0
        }
        
        public var needsEmbeddingUpdate: Bool {
            return self.status.isPending || (self.status.isActive && self.iterationsUntilEmbeddingUpdate <= 0)
        }
        
        public var rect: CGRect {
            return self.kalmanFilter.rect
        }
        
        // MARK: private properties
        
        private let configuration: TrackConfiguration
        private let kalmanFilter: VisualKF
        private let maxAppearanceCost: Float
        
        private var iterationsUntilEmbeddingUpdate: Int
        
        // MARK: constructors
        
        /// Track constructor
        /// - Parameter detection: `Detection` object that was assigned to this track
        /// - Parameter trackConfiguration: trackConfiguration of parent tracker
        /// - Parameter costConfiguration: costConfiguration of parent tracker
        /// - Throws: `TrackInitializationError.missingEmbedding` when `detection`'s embedding is `nil`
        public init(detection: Detection, trackConfiguration: TrackConfiguration, costConfiguration: CostConfiguration) throws {
            guard let embedding = detection.embedding else {
                throw TrackInitializationError.missingEmbedding
            }
            self.embedding = embedding
            self.kalmanFilter = VisualKF(initialObservation: detection.rect)
            self.maxAppearanceCost = costConfiguration.maxAppearanceCost
            self.averageAppearanceCost = costConfiguration.maxAppearanceCost / 2 // conservative estimate
            self.iterationsUntilEmbeddingUpdate = trackConfiguration.iterationsPerEmbeddingUpdate
            self.configuration = trackConfiguration
        }
        
        // MARK: public static methods
        
        static func == (lhs: Track, rhs: Track) -> Bool {
            return lhs.id == rhs.id // Compare properties
        }
        
        // MARK: public methods
        
        /// Run the Kalman filter's prediction step and record that another iteration has started.
        @inline(__always)
        func predict() {
            self.kalmanFilter.predict()
            self.iterationsUntilEmbeddingUpdate -= 1
        }
        
        /// Registers that this track was assigned a detection
        /// - Parameter detection: `Detection` object that was assigned to this track
        /// - Parameter costs: `Costs` object associated with the assignment.
        func registerHit(with detection: Detection, costs: Costs) {
            // register hit
            if !self.status.isActive {
                self.hits += 1
                
                // check if the track is ready for activation
                let threshold = (
                    self.status.isPending ?
                    self.configuration.confirmationThreshold :
                    self.configuration.activationThreshold
                )
                
                if self.hits >= threshold {
                    self.status = .active
                    self.hits = 0
                }
            } else {
                self.hits = 0
            }
            
            // update state
            self.kalmanFilter.update(measurement: detection.rect)
            
            /*  If the appearance cost was calculated then the detection's embedding must   *
             *  have also been computed. This is because the embedding is necessary to      *
             *  compute the appearance cost. Also, don't update embedding when inactive.    */
            if self.status.isInactive == false && costs.hasAppearance {
                self.updateEmbedding(detection: detection, appearanceCost: costs.appearance)
            }
            
            self.costs = costs
        }
        
        /// Registers that this track was not assigned a detection
        func registerMiss() {
            if self.status.isActive {
                self.hits -= 1
                if self.hits <= -self.configuration.deactivationThreshold {
                    self.status = .inactive
                    self.kalmanFilter.xVelocity = 0
                    self.kalmanFilter.yVelocity = 0
                    self.kalmanFilter.growthRate = 0
                    self.hits = 0
                } else {
                    self.kalmanFilter.xVelocity *= self.configuration.velocityDamping
                    self.kalmanFilter.yVelocity *= self.configuration.velocityDamping
                    self.kalmanFilter.growthRate *= self.configuration.growthDamping
                }
            } else {
                self.hits = 0
            }
        }
        
        /// Returns cosine distance between the feature embedding vectors
        /// - Parameter detection: `Detection` object whose appearance is being compared
        /// - Returns: cosine distance between this track's embedding vector and `detection`'s embedding vector
        @inline(__always)
        func cosineDistance(to detection: Detection) -> Float {
            if let detectionEmbedding = detection.embedding {
                return Utils.ML.cosineDistance(a: self.embedding, b: detectionEmbedding)
            }
            // return the maximum value of cosine distance
            return 2.0
        }
        
        /// Returns intersection over union
        /// - Parameter detection: `Detection` object that was assigned to this track
        /// - Returns: intersection over union of the track's rect with `detection`'s rect
        @inline(__always)
        func iou(with detection: Detection) -> Float {
            return Utils.iou(self.kalmanFilter.rect, detection.rect)
        }
        
        func hash(into hasher: inout Hasher) {
            hasher.combine(id)
        }
        
        /// Updates the embedding
        /// - Parameter detection: detection object that was assigned to this track
        /// - Parameter appearanceCost: appearance cost of the assignment
        func updateEmbedding(detection: Detection, appearanceCost: Float) {
            guard let newEmbedding = detection.embedding else { return }
            let alpha = self.configuration.embeddingAlpha * detection.confidence * exp(-appearanceCost / (self.averageAppearanceCost + 1e-10))
            self.averageAppearanceCost += (appearanceCost - self.averageAppearanceCost) * alpha
            Utils.ML.updateEMA(ema: self.embedding, with: newEmbedding, alpha: alpha)
            self.iterationsUntilEmbeddingUpdate = self.configuration.iterationsPerEmbeddingUpdate
        }
    }
    
    final class TrackConfiguration {
        public let confirmationThreshold: Int
        public let activationThreshold: Int
        public let deactivationThreshold: Int
        public let iterationsPerEmbeddingUpdate: Int
        public let embeddingAlpha: Float
        
        public let velocityDamping: Float
        public let growthDamping: Float
        
        init(confirmationThreshold: Int = 15,
             activationThreshold: Int = 2,
             deactivationThreshold: Int = 8,
             iterationsPerEmbeddingUpdate: Int = 5,
             embeddingAlpha: Float = 0.1,
             velocityDamping: Float = 0.5,
             growthDamping: Float = 0.1,
             dt: Float = 1.0 / 30.0)
        {
            self.confirmationThreshold = confirmationThreshold
            self.activationThreshold = activationThreshold
            self.deactivationThreshold = deactivationThreshold
            self.iterationsPerEmbeddingUpdate = iterationsPerEmbeddingUpdate
            self.embeddingAlpha = embeddingAlpha
            self.velocityDamping = pow(velocityDamping, dt)
            self.growthDamping = pow(growthDamping, dt)
        }
    }
}

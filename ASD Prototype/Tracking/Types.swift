//
//  Types.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/21/25.
//

import Foundation

extension Tracking {
    final class Costs {
        var iou: Float
        var appearance: Float
        var total: Float
        
        var hasIoU: Bool { self.iou != Float.infinity }
        var hasAppearance: Bool { self.appearance != Float.infinity }
        var hasTotal: Bool { self.total != Float.infinity }
        
        var string : String {
            let totalString: String = self.hasTotal ? "Cost = \(self.total)" : "Cost:"
            let iouString: String = self.hasIoU ? "\n\tIoU: \(self.iou)" : "\n"
            let appearanceString: String = self.hasAppearance ? "\n\tAppearance: \(self.appearance)" : "\n"
            return "\(totalString)\(iouString)\(appearanceString)"
        }
        
        init(iou: Float = Float.infinity, appearance: Float = Float.infinity, total: Float = Float.infinity) {
            self.iou = iou
            self.appearance = appearance
            self.total = total
        }
    }
    
    final class CostConfiguration {
        public let motionWeight: Float
        public let minIou: Float
        public let maxMotionCost: Float
        public let maxAppearanceCost: Float
        
        //old motion cost: 9.4877
        init(motionWeight: Float = 0.1, minIou: Float = 0.3, maxMotionCost: Float = Float.infinity, maxAppearanceCost: Float = 0.3) {
            self.motionWeight = motionWeight
            self.minIou = minIou
            self.maxMotionCost = maxMotionCost
            self.maxAppearanceCost = maxAppearanceCost
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

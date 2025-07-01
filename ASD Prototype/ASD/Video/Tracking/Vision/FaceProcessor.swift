//
//  FaceExtraction.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/19/25.
//

@preconcurrency import Vision
@preconcurrency import CoreML

import Foundation
import CoreImage
import Accelerate
import UIKit
import OrderedCollections
import ImageIO


extension ASD.Tracking {
    class FaceProcessor {
        // MARK: private properties
        
        private let detector: FaceDetector
        private let embedder: FaceEmbedder
        
        // MARK: public methods
        
        init (verbose: Bool = false,
              detectorConfidenceThreshold: Float = 0.5,
              embedderRequestLifespan: DispatchTimeInterval = .seconds(5),
              minReadyEmbedderRequests: Int = 8) {
            self.detector = FaceDetector(verbose: verbose, confidenceThreshold: detectorConfidenceThreshold)
            self.embedder = FaceEmbedder(verbose: verbose, requestLifespan: embedderRequestLifespan, minReadyRequests: minReadyEmbedderRequests)
        }
        
        public func detect(pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) -> OrderedSet<Detection> {
            let results = self.detector.detect(in: pixelBuffer, orientation: orientation)
            
            return OrderedSet(results.map {
                let rect = $0.boundingBox
                let box = CGRect(
                    x: rect.minX - rect.width * 0.2,
                    y: rect.minY,
                    width: rect.width * 1.4,
                    height: rect.height
                )
                return Detection(rect: box, confidence: Float($0.confidence))
            })
        }
        
        public func embed(pixelBuffer: CVPixelBuffer, faces detections: OrderedSet<Detection>, orientation: CGImagePropertyOrientation) {
            let results = self.embedder.embed(faces: detections.map{$0.rect}, in: pixelBuffer, orientation: orientation)
            
            for (i, result) in results.enumerated() {
                detections[i].embedding = result
            }
        }
    }
}

extension CVPixelBuffer: @unchecked @retroactive Sendable {}



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

extension Tracking {
    class FaceProcessor {
        private static let cropScale: CGFloat = 0.40
        private static let asdInputSize: CGSize = CGSize(width: 112, height: 112)
        
        // Vision and Core ML properties
        private let detector: FaceDetector
        private let embedder: FaceEmbedder
        
        private var pixelBuffer: CVPixelBuffer?
        
        // MARK: public methods
        
        init (verbose: Bool = false,
              detectorConfidenceThreshold: Float = 0.5,
              embedderRequestLifespan: DispatchTimeInterval = .seconds(5),
              minReadyEmbedderRequests: Int = 8) throws {
            let detector = FaceDetector(verbose: verbose, confidenceThreshold: detectorConfidenceThreshold)
            let embedder = FaceEmbedder(verbose: verbose, requestLifespan: embedderRequestLifespan, minReadyRequests: minReadyEmbedderRequests)
            (self.detector, self.embedder) = (detector, embedder)
        }
        
        public func loadFrame(from sampleBuffer: CMSampleBuffer) {
            if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                self.pixelBuffer = pixelBuffer
            }
        }
        
        public func detect() -> OrderedSet<Detection> {
            guard let pixelBuffer = self.pixelBuffer else { return [] }
            let results = self.detector.detect(in: pixelBuffer)
            
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
        
        public func embed(faces detections: OrderedSet<Detection>) {
            guard let pixelBuffer = self.pixelBuffer else { return }
            let results = self.embedder.embed(faces: detections.map{$0.rect}, in: pixelBuffer)
            
            for (i, result) in results.enumerated() {
                detections[i].embedding = result
            }
        }
    }
}

extension CVPixelBuffer: @unchecked @retroactive Sendable {}



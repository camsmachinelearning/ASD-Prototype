//
//  ASD.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/23/25.
//

import Foundation
@preconcurrency import CoreML
@preconcurrency import AVFoundation
import Vision

extension ASD {
    final actor ASD {
        private let audioEncoder: AudioEncoder
        private let videoProcessor: VideoProcessor
        private var lastAudioTime: Double
        
        private nonisolated let model: ASDVideoModel = {
            let config = MLModelConfiguration()
            // Explicitly avoid the ANE
            config.computeUnits = .cpuAndGPU

            do {
                return try ASDVideoModel(configuration: config)
            } catch {
                fatalError("Could not load model: \\(error)")
            }
        }()
        
        init(atTime time: Double, backtrackFrames: Int = 5, audioBufferPadding: Int = 25, videoBufferPadding: Int = 12) {
            self.videoProcessor = .init(atTime: time, backtrackFrames: backtrackFrames, videoBufferPadding: videoBufferPadding)
            self.audioEncoder = .init(atTime: time, frontPadding: backtrackFrames, backPadding: audioBufferPadding)
            self.lastAudioTime = time
        }
        
        public func updateAudio(audioSample sampleBuffer: CMSampleBuffer) async {
//            let start = Date()
            
            let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
            let signal = resampleAudioToFloat32(from: sampleBuffer, to: 16_000)
            await audioEncoder.update(atTime: time, from: signal)
            self.lastAudioTime = time
            
//            let end = Date()
//            let elapsed = end.timeIntervalSince(start)  // in seconds (Double)
//            print("AudioUpdate: \(elapsed * 1000) ms")
        }
        
        public func update(videoSample sampleBuffer: CMSampleBuffer, connection: AVCaptureConnection) async throws -> [SpeakerData]? {
            let start = Date()
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
//            let lastAvailibleTime = min(time, self.lastAudioTime)
            
            async let _ = self.videoProcessor.updateVideos(atTime: time, from: pixelBuffer, connection: connection)
            //async let audioEmbedAsync = try self.audioEncoder.encode(atTime: lastAvailibleTime)
            let videoInputs = await self.videoProcessor.getFrames(at: -1)
            
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
            
            var scores: [UUID : MLMultiArray] = [:]
            for (id, videoInput) in videoInputs {
                let input = ASDVideoModelInput(videoInput: videoInput)
                let output = try await self.model.prediction(input: input).scores
                scores[id] = output
            }
            let res = await self.videoProcessor.updateScores(atTime: time, with: scores)
            let end = Date()
            let elapsed = end.timeIntervalSince(start)  // in seconds (Double)
            print("VideoUpdate: \(elapsed * 1000) ms")
            return res
        }
    }
}

extension ASDVideoModelOutput : @unchecked Sendable {}
extension ASDVideoModel: @unchecked Sendable {}

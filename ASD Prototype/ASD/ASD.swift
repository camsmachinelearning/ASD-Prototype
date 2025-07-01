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
    final class ASD {
        public private(set) var speakers: [SpeakerData]
        
        //private let audioEncoder: AudioEncoder
        private let videoProcessor: VideoProcessor
        //private var lastAudioTime: Double
        private var lastVideoTime: Double
        private var frameSkipCounter: Int = 0
        
        private let onFused: @Sendable ([SpeakerData]) async -> Void
        private let modelPool: Utils.ML.ModelPool<ASDVideoModel>
        
        init(atTime time: Double, onFused: @Sendable @escaping ([SpeakerData]) async -> Void, numModels: Int = 6, backtrackFrames: Int = 25, audioBufferPadding: Int = 25, videoBufferPadding: Int = 24) {
            self.videoProcessor = .init(atTime: time, backtrackFrames: backtrackFrames, videoBufferPadding: videoBufferPadding)
            //self.audioEncoder = .init(atTime: time, frontPadding: backtrackFrames, backPadding: audioBufferPadding)
            //self.lastAudioTime = time
            self.lastVideoTime = time
            self.frameSkipCounter = 0
            self.speakers = []
            self.onFused = onFused
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndGPU
            self.modelPool = try! .init(count: numModels) {
                try .init(configuration: configuration)
            }
        }
        
//        public func updateAudio(audioSample sampleBuffer: CMSampleBuffer) {
////            let start = Date()
//            let audioEncoder = self.audioEncoder
//            let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
//            let signal = resampleAudioToFloat32(from: sampleBuffer, to: 16_000)
//            
//            Task.detached {
//                await audioEncoder.update(atTime: time, from: signal)
//            }
//            
//            self.lastAudioTime = time
//            
////            let end = Date()
////            let elapsed = end.timeIntervalSince(start)  // in seconds (Double)
////            print("AudioUpdate: \(elapsed * 1000) ms")
//        }
        
        public func update(videoSample sampleBuffer: CMSampleBuffer, connection: AVCaptureConnection) throws {
            let start = Date()
            
            guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            let time = sampleBuffer.presentationTimeStamp.seconds
            
            // determine if we skip this frame to update ASD
            self.frameSkipCounter += 1
            let isVideoUpdate = self.frameSkipCounter < 6
            if isVideoUpdate == false {
                self.frameSkipCounter = 0
                self.lastVideoTime = time
                print("skipping video update")
            } else {
                print("updating video")
            }
            
            let asdTime = self.lastVideoTime //min(self.lastVideoTime, self.lastAudioTime)
            let videoProcessor = self.videoProcessor
            //let audioEncoder = self.audioEncoder
            let callback = self.onFused
            let modelPool = self.modelPool
            
            Task.detached {
                if isVideoUpdate {
                    await videoProcessor.updateVideos(atTime: time, from: pixelBuffer, connection: connection, skip: false)
                    CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                    let res = await videoProcessor.getSpeakers(atTime: time)
                    print(res)
                    await callback(res)
                } else {
                    async let _ = videoProcessor.updateVideos(atTime: time, from: pixelBuffer, connection: connection, skip: true)
                    //async let audioEmbedAsync = try audioEncoder.encode(atTime: asdTime)
                    //let (videoInputs, audioEmbed) = await (videoProcessor.getFrames(atTime: asdTime), try audioEmbedAsync)
                    let videoInputs = await videoProcessor.getFrames(atTime: asdTime)
                    CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
                    
                    let scores: [UUID: MLMultiArray] = try await withThrowingTaskGroup(of: (UUID, MLMultiArray).self) { group in
                        for (id, videoInput) in videoInputs {
                            group.addTask {
                                //let input = ASDModelInput(audioEmbedding: audioEmbed, videoInput: videoInput)
                                let input = ASDVideoModelInput(videoInput: videoInput)
                                let scores = try await modelPool.withModel { model in
                                    try model.prediction(input: input).scores
                                }
                                return (id, scores)
                            }
                        }

                        var results: [UUID: MLMultiArray] = [:]

                        for try await (id, scores) in group {
                            results[id] = scores
                        }

                        return results
                    }
                    
                    let res = await videoProcessor.updateScores(atTime: asdTime, with: scores)
                    await callback(res)
                }
                let end = Date()
                let elapsed = end.timeIntervalSince(start)  // in seconds (Double)
                print("VideoUpdate: \(elapsed * 1000) ms")
            }
        }
    }
}

extension ASDVideoModelOutput : @unchecked Sendable {}
extension ASDVideoModel: @unchecked Sendable {}

extension ASDVideoModel: MLWrapper {
    typealias Input = ASDVideoModelInput
    typealias Output = ASDVideoModelOutput
}

//
//  VideoHandler.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/29/25.
//

import Foundation
import CoreML
import AVFoundation

import UIKit
import CoreML
import ImageIO
import MobileCoreServices

/// Saves a 4D MLMultiArray ([1, T, H, W]) as an animated GIF in the app's Documents directory.
/// - Parameters:
///   - array: The MLMultiArray with shape [1, T, H, W] (Float32-compatible).
///   - fileName: Name for the GIF file (e.g. "output.gif").
func saveMultiArrayAsGIF(_ array: MLMultiArray, fileName: String) {
    // Validate shape
    let shape = array.shape.map { $0.intValue }
    guard shape.count == 4, shape[0] == 1 else {
        print("❌ Expected shape [1, T, H, W], got \(shape)")
        return
    }
    let T = shape[1], H = shape[2], W = shape[3]

    // Prepare frames as UIImage
    var frames: [UIImage] = []
    for t in 0..<T {
        // Extract 2D slice and normalize
        var plane = [Float](repeating: 0, count: H * W)
        var minv = Float.greatestFiniteMagnitude
        var maxv = -Float.greatestFiniteMagnitude
        for y in 0..<H {
            for x in 0..<W {
                let v = array[[0, t, y, x] as [NSNumber]].floatValue
                plane[y * W + x] = v
                minv = min(minv, v)
                maxv = max(maxv, v)
            }
        }
        let range = (maxv - minv) != 0 ? (maxv - minv) : 1
        let pixels = plane.map { UInt8(clamping: Int((($0 - minv) / range) * 255)) }

        // Create CGImage from grayscale data
        guard let cfData = CFDataCreate(nil, pixels, W * H) else { continue }
        let provider = CGDataProvider(data: cfData)!
        let cgImage = CGImage(
            width: W, height: H,
            bitsPerComponent: 8, bitsPerPixel: 8,
            bytesPerRow: W,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent
        )!
        frames.append(UIImage(cgImage: cgImage))
    }

    // Determine output URL in Documents
    let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
    let gifURL = docs.appendingPathComponent(fileName)

    // Create GIF
    guard let dest = CGImageDestinationCreateWithURL(gifURL as CFURL, kUTTypeGIF, frames.count, nil) else {
        print("❌ Could not create GIF destination at \(gifURL.path)")
        return
    }

    // Container-level properties (must be set before adding frames)
    let gifProperties = [
        kCGImagePropertyGIFDictionary: [
            kCGImagePropertyGIFLoopCount: 0  // loop forever
        ]
    ] as CFDictionary
    CGImageDestinationSetProperties(dest, gifProperties)

    // Per-frame properties
    let frameProperties = [
        kCGImagePropertyGIFDictionary: [
            kCGImagePropertyGIFDelayTime: 0.1  // seconds per frame
        ]
    ] as CFDictionary

    // Add frames
    for image in frames {
        guard let cg = image.cgImage else { continue }
        CGImageDestinationAddImage(dest, cg, frameProperties)
    }

    // Finalize
    if CGImageDestinationFinalize(dest) {
        print("✅ GIF saved to: \(gifURL.path)")
    } else {
        print("❌ Failed to write GIF.")
    }
}

// Usage Example:
// Assuming `myArray` is your 1×T×H×W MLMultiArray:
// saveMultiArrayAsGIF(myArray, fileName: "my_multiarray.gif")



extension ASD {
    actor VideoProcessor {
        struct VideoTrack {
            let videoBuffer: VideoBuffer
            let scoreBuffer: ScoreBuffer
            var track: Tracking.Tracker.SendableTrack
            
            var lastFrameWriteTime: Double
            var writes: Int = 0
            
            
            init(atTime time: Double, track: Tracking.Tracker.SendableTrack, backtrackFrames: Int, videoBufferPadding: Int = 25, scoreBufferCapacity: Int = 53) {
                self.lastFrameWriteTime = time
                self.track = track
                self.videoBuffer = .init(frontPadding: backtrackFrames, backPadding: videoBufferPadding)
                self.scoreBuffer = .init(atTime: time, frontPadding: backtrackFrames, capacity: scoreBufferCapacity)
            }
            
            mutating func updateVideo(atTime time: Double, from pixelBuffer: CVPixelBuffer, with track: Tracking.Tracker.SendableTrack) {
                self.videoBuffer.write(from: pixelBuffer, croppedTo: track.rect)
                self.lastFrameWriteTime = time
                self.track = track
                self.writes += 1
            }
        }
        
        // MARK: Public properties
        public private(set) var speakers: [UUID: VideoTrack]
        public var lastVideoTime: Double { self.videoTimestamps.lastWriteTime }
        public var lastScoreTime: Double { self.scoreTimestamps.lastWriteTime }
        
        // MARK: Private properties
        private let tracker: Tracking.Tracker
        private let backtrackFrames: Int
        private let videoBufferPadding: Int
        private let videoTimestamps: Utils.TimestampBuffer
        private let scoreTimestamps: Utils.TimestampBuffer
        
        // MARK: Constructor
        init(atTime time: Double, backtrackFrames: Int, videoBufferPadding: Int = 12) {
            self.videoBufferPadding = videoBufferPadding
            self.backtrackFrames = backtrackFrames
            self.tracker = .init(faceProcessor: .init())
            self.videoTimestamps = .init(atTime: time, capacity: 25 + videoBufferPadding + backtrackFrames)
            self.scoreTimestamps = .init(atTime: time, capacity: 53)
            self.speakers = [:]
        }
        
        // MARK: Updater methods
        public func updateVideos(atTime time: Double, from pixelBuffer: CVPixelBuffer, connection: AVCaptureConnection) {
            // get tracks
            let orientation = Utils.Images.cgImageOrientation(fromRotationAngle: connection.videoRotationAngle,
                                                              mirrored: connection.isVideoMirrored)
            let tracks = tracker.update(pixelBuffer: pixelBuffer, orientation: orientation)
            
            // update speakers
            for track in tracks where track.rect.width.isNaN == false {
                self.speakers[track.id, default: self.makeSpeaker(atTime: time, track: track)]
                    .updateVideo(atTime: time, from: pixelBuffer, with: track)
            }
            
            self.speakers = self.speakers.filter { _, speaker in
                speaker.lastFrameWriteTime >= time
            }
            
            // update timestamps
            self.videoTimestamps.write(atTime: time)
        }
        
        public func updateScores(atTime time: Double, with scores: [UUID : MLMultiArray]) -> [SpeakerData] {
            for (id, score) in scores {
                self.speakers[id]?.scoreBuffer.write(from: score)
            }
            self.videoTimestamps.write(atTime: time)
            return self.speakers.values.map { speaker in
                    .init(track: speaker.track, score: speaker.scoreBuffer.read(at: -1))
            }
        }
        
        // MARK: Getter methods
        public func getFrames(atTime time: Double?) -> [UUID : MLMultiArray] {
            let index = self.videoTimestamps.indexOf(time ?? self.lastVideoTime)
            return Dictionary(uniqueKeysWithValues: self.speakers.map { id, speaker in
                return (id, speaker.videoBuffer.read(at: index))
            })
        }
        
        // MARK: Getter methods
        public func getFrames(at index: Int = -1) -> [UUID : MLMultiArray] {
            return Dictionary(uniqueKeysWithValues: self.speakers.map { id, speaker in
                return (id, speaker.videoBuffer.read(at: index))
            })
        }
        
        public func getScores(atTime time: Double?) -> [UUID : Float] {
            let index = self.scoreTimestamps.indexOf(time ?? self.lastScoreTime)
            return Dictionary(uniqueKeysWithValues: self.speakers.map { id, speaker in
                (id, speaker.scoreBuffer.read(at: index))
            })
        }
        
        public func getSpeakers(atTime time: Double) -> [SpeakerData] {
            let index = self.scoreTimestamps.indexOf(time)
            return self.speakers.values.map { speaker in
                    .init(track: speaker.track, score: speaker.scoreBuffer.read(at: index))
            }
        }
        
        // MARK: Private helpers
        @inline(__always)
        private func makeSpeaker(atTime time: Double, track: Tracking.Tracker.SendableTrack) -> VideoTrack {
            return .init(atTime: time,
                         track: track,
                         backtrackFrames: self.backtrackFrames,
                         videoBufferPadding: self.videoBufferPadding,
                         scoreBufferCapacity: 25)
        }
    }
}

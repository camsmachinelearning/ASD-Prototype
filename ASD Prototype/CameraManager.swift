// =================================================================
// FILE 1: CameraManager.swift
// Create a new Swift file and name it CameraManager.swift
// =================================================================
// This class is the heart of our app. It handles setting up the camera,
// processing video frames, and running the Core ML model via Vision.
// IMPORTANT: This class has no knowledge of the UI. It only deals with
// camera data and provides normalized results (coordinates from 0.0 to 1.0).

import AVFoundation
import Vision
import UIKit
import SwiftUI

extension Tracking {
    class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
        
        // This allows the CameraPreview view to reactively update when the session is ready.
        @Published var captureSession: AVCaptureSession?
        
        // Published properties to update the SwiftUI view
        @Published public private(set) var detections: [Face] = []
        
        @Published public private(set) var videoSize: CGSize
        
        // AVFoundation properties
        private let videoOutput = AVCaptureVideoDataOutput()
        private let sessionQueue = DispatchQueue(label: "com.facedetector.sessionQueue")
        
        // Vision and Core ML properties
        private var tracker: Tracker?
        
        override init() {
            self.videoSize = CGSize(width: 720, height: 1280)
            super.init()
            // Asynchronously check permissions and then set up the session
            sessionQueue.async {
                let faceProcessor = try? FaceProcessor()
                self.tracker = Tracker(faceProcessor: faceProcessor!)
                self.checkPermissionsAndSetup()
            }
        }
        
        // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
        
        func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
            self.tracker?.update(sampleBuffer: sampleBuffer)
            let res = self.tracker?.activeFaces ?? []
            
            Task { @MainActor [res] in
                self.detections = res
            }
        }
        
        // Public methods to control session from the UI. These are useful for app lifecycle events.
        func startSession() {
            sessionQueue.async {
                if self.captureSession?.isRunning == false {
                    self.captureSession?.startRunning()
                }
            }
        }
        
        func stopSession() {
            sessionQueue.async {
                if self.captureSession?.isRunning == true {
                    self.captureSession?.stopRunning()
                }
            }
        }
        
        // MARK: - AVFoundation Camera Setup
        
        private func checkPermissionsAndSetup() {
            switch AVCaptureDevice.authorizationStatus(for: .video) {
            case .authorized:
                setupCaptureSession()
            case .notDetermined:
                AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                    if granted {
                        self?.setupCaptureSession()
                    } else {
                        print("Camera access was denied.")
                    }
                }
            default:
                print("Camera access is restricted or denied.")
            }
        }
        
        private func setupCaptureSession() {
            // This method should only be called from the sessionQueue
            
            let session = AVCaptureSession()
            session.sessionPreset = .hd1280x720
            
            //        guard let device = AVCaptureDevice.default(for: .video) else { return }
            //        let resolution = CMVideoFormatDescriptionGetDimensions(device.activeFormat.formatDescription)
            //        DispatchQueue.main.async {
            //            self.videoSize = CGSize(width: CGFloat(resolution.width), height: CGFloat(resolution.height))
            //        }
            
            guard let captureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else {
                print("Error: No back camera found.")
                return
            }
            
            do {
                let input = try AVCaptureDeviceInput(device: captureDevice)
                if session.canAddInput(input) {
                    session.addInput(input)
                }
            } catch {
                print("Error setting up camera input: \(error)")
                return
            }
            
            videoOutput.setSampleBufferDelegate(self, queue: sessionQueue)
            videoOutput.alwaysDiscardsLateVideoFrames = true
            videoOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            
            if session.canAddOutput(videoOutput) {
                session.addOutput(videoOutput)
            } else {
                print("Error: Could not add video output.")
                return
            }
            
            if let connection = videoOutput.connection(with: .video) {
                // Use the new API on iOS 17 and later
                if #available(iOS 17.0, *) {
                    // To set portrait orientation, we check for and set a 90-degree rotation.
                    if connection.isVideoRotationAngleSupported(90) {
                        connection.videoRotationAngle = 90
                    }
                } else {
                    // Fallback for earlier iOS versions
                    if connection.isVideoOrientationSupported {
                        connection.videoOrientation = .portrait
                    }
                }
            }
            
            // Start the session on the background queue immediately after configuration.
            session.startRunning()
            
            // Publish the now-running session to the main thread.
            
            Task { @MainActor in
                self.captureSession = session
            }
        }
    }
}

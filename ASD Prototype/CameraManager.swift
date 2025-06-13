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

// A custom identifiable struct for our bounding boxes.
// This is best practice for use in SwiftUI's ForEach.
struct DetectedBox: Identifiable {
    let id = UUID()
    let rect: CGRect // Normalized coordinates (0-1)
}

class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    
    // This allows the CameraPreview view to reactively update when the session is ready.
    @Published var captureSession: AVCaptureSession?
    
    // Published properties to update the SwiftUI view
    @Published var boundingBoxes: [DetectedBox] = []
    
    // AVFoundation properties
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.facedetector.sessionQueue")
    
    // Vision and Core ML properties
    private var visionModel: VNCoreMLModel?
    
    override init() {
        super.init()
        // Asynchronously check permissions and then set up the session
        sessionQueue.async {
            self.setupVision()
            self.checkPermissionsAndSetup()
        }
    }
    
    // MARK: - Core ML and Vision Setup
    
    private func setupVision() {
        // Ensure the model name matches the file you added to the project.
        guard let modelURL = Bundle.main.url(forResource: "yolov11n-face", withExtension: "mlmodelc") else {
            print("Critical Error: Core ML model file ('yolov8s.mlmodelc') not found in bundle.")
            return
        }
        
        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            self.visionModel = visionModel
        } catch {
            print("Error loading or creating Vision model: \(error)")
        }
    }
    
    // Completion handler for the Vision request. This is where we get the results.
    private func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let results = request.results as? [VNRecognizedObjectObservation] {
            // Filter results for confidence and map to our custom struct
            let detectedBoxes = results.filter { $0.confidence > 0.5 }.map {
                DetectedBox(rect: $0.boundingBox)
            }
            
            // Update the UI on the main thread
            DispatchQueue.main.async {
                self.boundingBoxes = detectedBoxes
            }
        } else {
            // If no objects are found, clear the boxes
            DispatchQueue.main.async {
                self.boundingBoxes = []
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
        DispatchQueue.main.async {
            self.captureSession = session
        }
    }
    
    // MARK: - AVCaptureVideoDataOutputSampleBufferDelegate
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let visionModel = self.visionModel else {
            return
        }
        
        let recognitionRequest = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
        recognitionRequest.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        
        do {
            try handler.perform([recognitionRequest])
        } catch {
            print("Failed to perform Vision request: \(error)")
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
}

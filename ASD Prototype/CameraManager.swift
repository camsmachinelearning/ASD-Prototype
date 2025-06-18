//
//  CameraManager.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/12/25.
//

import AVFoundation
import Vision
import UIKit
import SwiftUI
import LANumerics

struct DetectedBox: Identifiable {
    let id = UUID()
    let rect: CGRect
    let boxColor: UIColor // New property for color
}

class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var captureSession: AVCaptureSession?
    @Published var boundingBoxes: [DetectedBox] = []

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.facedetector.sessionQueue")
    
    private var faceDetectionModel: VNCoreMLModel?
    private var faceEmbeddingModel: VNCoreMLModel?
    
    private var tracker = Tracker() // Instantiate the Tracker
    private let coreMotionManager = CoreMotionManager() // Not directly used in this version, but kept for context

    override init() {
        super.init()
        sessionQueue.async {
            self.setupVision()
            self.checkPermissionsAndSetup()
        }
    }
    
    private func setupVision() {
        // Face Detector
        do {
            let faceDetectorWrapper = try YOLOv11n_face(configuration: MLModelConfiguration())
            let visionModel = try VNCoreMLModel(for: faceDetectorWrapper.model)
            self.faceDetectionModel = visionModel
        } catch {
            print("Error loading Face Detection model: \(error)")
        }
        
        // Face Embedding
        do {
            let faceEmbeddingWrapper = try MobileFaceNet(configuration: MLModelConfiguration()) // Correctly load MobileFaceNet
            let visionModel = try VNCoreMLModel(for: faceEmbeddingWrapper.model)
            self.faceEmbeddingModel = visionModel
        } catch {
            print("Error loading Face Embedding model: \(error)")
        }
    }
    
    private func checkPermissionsAndSetup() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            setupCaptureSession()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    self?.setupCaptureSession()
                }
            }
        default:
            print("Camera access denied.")
        }
    }
    
    private func setupCaptureSession() {
        let session = AVCaptureSession()
        session.sessionPreset = .hd1280x720
        
        guard let captureDevice = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) else { return }
        
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
        if session.canAddOutput(videoOutput) {
            session.addOutput(videoOutput)
        }
        
        session.startRunning()
        DispatchQueue.main.async {
            self.captureSession = session
        }
    }
    
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer),
              let faceDetectorModel = self.faceDetectionModel else {
            return
        }
        
        Task {
            // 1. Create and perform the face detection request
            let faceDetectionRequest = VNCoreMLRequest(model: faceDetectorModel)
            faceDetectionRequest.imageCropAndScaleOption = .scaleFill
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
            
            do {
                try await handler.perform([faceDetectionRequest])
                
                var newDetectionsForTracker: [Detection] = []
                var boxesToDraw: [DetectedBox] = []
                
                if let observations = faceDetectionRequest.results as? [VNRecognizedObjectObservation] {
                    // Filter raw detections by confidence and add them to boxesToDraw as red
                    let confidentDetections = observations.filter { $0.confidence >= 0.5 }
                    for detection in confidentDetections {
                        boxesToDraw.append(DetectedBox(rect: detection.boundingBox, boxColor: .red))
                    }
                    
                    if !confidentDetections.isEmpty {
                        let embeddings = await self.runEmbeddingConcurrently(on: pixelBuffer, for: confidentDetections)
                        
                        // Combine confident detections with their corresponding embeddings for the tracker
                        for (index, detection) in confidentDetections.enumerated() {
                            if index < embeddings.count {
                                let newDetection = Detection(box: detection.boundingBox, embedding: embeddings[index], confidence: detection.confidence)
                                newDetectionsForTracker.append(newDetection)
                            }
                        }
                    }
                    
                    // Update the tracker with the new detections
                    self.tracker.update(detections: newDetectionsForTracker)
                    
                    // Add tracked faces to boxesToDraw as green
                    for track in self.tracker.activeTracks {
                        boxesToDraw.append(DetectedBox(rect: track.kalmanFilter.rect, boxColor: .green))
                    }
                }
                
                // Update boundingBoxes for UI
                DispatchQueue.main.async {
                    self.boundingBoxes = boxesToDraw
                }
            } catch {
                print("Failed to perform Vision request: \(error)")
            }
        }
    }
    
    /// Creates a TaskGroup to run the embedding model on multiple detections concurrently.
        private func runEmbeddingConcurrently(on pixelBuffer: CVPixelBuffer, for detections: [VNRecognizedObjectObservation]) async -> [MLMultiArray] {
            guard self.faceEmbeddingModel != nil else { return [] }

            return await withTaskGroup(of: MLMultiArray?.self, returning: [MLMultiArray].self) { group in
                for detection in detections {
                    // Add a new async task to the group for each detection.
                    // The group runs these tasks in parallel.
                    group.addTask {
                        return await self.getEmbedding(on: pixelBuffer, for: detection)
                    }
                }
                
                var collectedEmbeddings: [MLMultiArray] = []
                // Await each task's result as it completes and collect them.
                for await embedding in group {
                    if let embedding = embedding {
                        collectedEmbeddings.append(embedding)
                    }
                }
                return collectedEmbeddings
            }
        }

        /// Runs the face embedding model for a single detection.
        private func getEmbedding(on pixelBuffer: CVPixelBuffer, for detection: VNRecognizedObjectObservation) async -> MLMultiArray? {
            guard let faceEmbedderModel = self.faceEmbeddingModel else { return nil }

            let embeddingRequest = VNCoreMLRequest(model: faceEmbedderModel)
            embeddingRequest.regionOfInterest = detection.boundingBox
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
            
            do {
                try await handler.perform([embeddingRequest])
                if let results = embeddingRequest.results as? [VNCoreMLFeatureValueObservation],
                   let firstResult = results.first {
                    return firstResult.featureValue.multiArrayValue
                }
            } catch {
                print("Failed to run embedding model for a detection: \(error)")
            }
            return nil
        }

    
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

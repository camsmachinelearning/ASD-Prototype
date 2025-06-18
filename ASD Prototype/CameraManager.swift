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
    let isPrediction: Bool // To distinguish between real detections and predictions
}



class CameraManager: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var captureSession: AVCaptureSession?
    @Published var boundingBoxes: [DetectedBox] = []

    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.facedetector.sessionQueue")
    
    private var faceDetectionModel: VNCoreMLModel?
    private var faceEmbeddingModel: VNCoreMLModel?
    
    private var trackedObjects: [KalmanFilter] = []
    private let coreMotionManager = CoreMotionManager()

    private var framesSinceLastDetection = 0
    private let maxFramesToPredict = 15 // Predict for 0.5 seconds at 30fps

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
            let faceDetectorWrapper = try YOLOv11n_face(configuration: MLModelConfiguration())
            let visionModel = try VNCoreMLModel(for: faceDetectorWrapper.model)
            self.faceDetectionModel = visionModel
        } catch {
            print("Error loading Face Detection model: \(error)")
        }
    }
    
    private func visionRequestDidComplete(request: VNRequest, error: Error?) {
//        if let results = request.results as? [VNRecognizedObjectObservation], let detections = results.filter {pred in pred.confidence > 0.5 } {
//            let detectedRect = bestResult[].boundingBox
//            framesSinceLastDetection = 0
//            
//            if kalmanFilter == nil {
//                kalmanFilter = KalmanFilter(initialObservation: detectedRect)
//            } else {
//                let measurement = Vector<Float>([
//                    Float(detectedRect.midX),
//                    Float(detectedRect.midY),
//                    Float(detectedRect.width),
//                    Float(detectedRect.height)
//                ])
//                kalmanFilter?.update(measurement: measurement)
//            }
//            
//            if let filteredRect = kalmanFilter?.predictedRect {
//                DispatchQueue.main.async {
//                    self.boundingBoxes = [DetectedBox(rect: filteredRect, isPrediction: false)]
//                }
//            }
//        } else {
//            framesSinceLastDetection += 1
//            if framesSinceLastDetection <= maxFramesToPredict, let kf = kalmanFilter {
//                kf.predict()
//                // Placeholder for IMU data integration
//                // You would use `coreMotionManager.rotationRate` to adjust the prediction
//                let predictedRect = kf.predictedRect
//                
//                // Stop motion if predicted to be off-camera
//                if isOffscreen(rect: predictedRect) {
//                    kalmanFilter = nil // Stop tracking
//                    DispatchQueue.main.async { self.boundingBoxes = [] }
//                } else {
//                    DispatchQueue.main.async {
//                        self.boundingBoxes = [DetectedBox(rect: predictedRect, isPrediction: true)]
//                    }
//                }
//            } else {
//                kalmanFilter = nil
//                DispatchQueue.main.async {
//                    self.boundingBoxes = []
//                }
//            }
//        }
    }
    
    private func isOffscreen(rect: CGRect) -> Bool {
        return rect.maxX < 0 || rect.minX > 1 || rect.maxY < 0 || rect.minY > 1
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
                
                if let detections = (faceDetectionRequest.results as? [VNRecognizedObjectObservation])?.filter({ $0.confidence > 0.5 }), !detections.isEmpty {
                    let embeddings = await self.runEmbeddingConcurrently(on: pixelBuffer, for: detections)

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

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
    
    private var visionModel: VNCoreMLModel?
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
        guard let modelURL = Bundle.main.url(forResource: "yolov11n-face", withExtension: "mlmodelc") else {
            print("Error: Model file not found")
            return
        }
        
        do {
            let visionModel = try VNCoreMLModel(for: MLModel(contentsOf: modelURL))
            self.visionModel = visionModel
        } catch {
            print("Error loading Vision model: \(error)")
        }
    }
    
    private func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let results = request.results as? [VNRecognizedObjectObservation], let bestResult = results.first(where: { $0.confidence > 0.5 }) {
            let detectedRect = bestResult.boundingBox
            framesSinceLastDetection = 0
            
            if kalmanFilter == nil {
                kalmanFilter = KalmanFilter(initialObservation: detectedRect)
            } else {
                let measurement = Vector<Float>([
                    Float(detectedRect.midX),
                    Float(detectedRect.midY),
                    Float(detectedRect.width),
                    Float(detectedRect.height)
                ])
                kalmanFilter?.update(measurement: measurement)
            }
            
            if let filteredRect = kalmanFilter?.predictedRect {
                DispatchQueue.main.async {
                    self.boundingBoxes = [DetectedBox(rect: filteredRect, isPrediction: false)]
                }
            }
        } else {
            framesSinceLastDetection += 1
            if framesSinceLastDetection <= maxFramesToPredict, let kf = kalmanFilter {
                kf.predict()
                // Placeholder for IMU data integration
                // You would use `coreMotionManager.rotationRate` to adjust the prediction
                let predictedRect = kf.predictedRect
                
                // Stop motion if predicted to be off-camera
                if isOffscreen(rect: predictedRect) {
                    kalmanFilter = nil // Stop tracking
                    DispatchQueue.main.async { self.boundingBoxes = [] }
                } else {
                    DispatchQueue.main.async {
                        self.boundingBoxes = [DetectedBox(rect: predictedRect, isPrediction: true)]
                    }
                }
            } else {
                kalmanFilter = nil
                DispatchQueue.main.async {
                    self.boundingBoxes = []
                }
            }
        }
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
              let visionModel = self.visionModel else {
            return
        }
        
        let request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
        request.imageCropAndScaleOption = .scaleFill
        
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        try? handler.perform([request])
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

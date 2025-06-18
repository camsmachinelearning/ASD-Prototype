//
//  ContentView.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/12/25.
//

import SwiftUI
import Vision
import AVFoundation

class DrawingView: UIView {
    var boundingBoxes: [DetectedBox] = [] {
        didSet {
            DispatchQueue.main.async {
                self.setNeedsDisplay()
            }
        }
    }
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        guard let context = UIGraphicsGetCurrentContext() else { return }
        
        // Blackout effect
        let blackoutPath = CGMutablePath()
        blackoutPath.addRect(bounds)
        
        for box in boundingBoxes {
            let drawingRect = VNImageRectForNormalizedRect(box.rect, Int(bounds.width), Int(bounds.height))
            // The VNImageRectForNormalizedRect function already returns a rectangle with top-left origin,
            // so no further flipping is needed.
            let correctedRect = drawingRect
            
            blackoutPath.addRect(correctedRect)
        }
        
        context.addPath(blackoutPath)
        context.setFillColor(UIColor.black.withAlphaComponent(0.7).cgColor)
        context.fillPath(using: .evenOdd)

        // Draw bounding box outlines
        for box in boundingBoxes {
            let drawingRect = VNImageRectForNormalizedRect(box.rect, Int(bounds.width), Int(bounds.height))
            // The VNImageRectForNormalizedRect function already returns a rectangle with top-left origin,
            // so no further flipping is needed.
            let correctedRect = drawingRect
            
            // Set color based on the boxColor property
            context.setStrokeColor(box.boxColor.cgColor)
            context.setLineWidth(3)
            context.stroke(correctedRect)
        }
    }
}

struct CameraPreview: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        
        let previewLayer = AVCaptureVideoPreviewLayer()
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        let drawingView = DrawingView(frame: view.bounds)
        drawingView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(drawingView)
        
        context.coordinator.previewLayer = previewLayer
        context.coordinator.drawingView = drawingView
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        if context.coordinator.previewLayer?.session == nil, let session = cameraManager.captureSession {
            context.coordinator.previewLayer?.session = session
        }
        
        context.coordinator.previewLayer?.frame = uiView.bounds
        context.coordinator.drawingView?.frame = uiView.bounds
        context.coordinator.drawingView?.boundingBoxes = cameraManager.boundingBoxes
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
        var drawingView: DrawingView?
    }
}

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    
    var body: some View {
        CameraPreview(cameraManager: cameraManager)
            .ignoresSafeArea()
            .onAppear(perform: cameraManager.startSession)
            .onDisappear(perform: cameraManager.stopSession)
    }
}

// =================================================================
// FILE 2: ContentView.swift
// Replace the contents of the default ContentView.swift
// =================================================================
// This file now contains the main SwiftUI view AND the UIKit views
// responsible for rendering the camera feed and bounding boxes correctly.

import SwiftUI
import Vision
import AVFoundation

// A custom UIView for drawing the bounding boxes. This is more reliable
// than SwiftUI for this use case because its coordinate system is
// directly tied to the camera preview layer's frame.
class DrawingView: UIView {
    var boundingBoxes: [CGRect] = [] {
        // When this property is set, redraw the view.
        didSet {
            // Must be called on the main thread.
            DispatchQueue.main.async {
                self.setNeedsDisplay()
            }
        }
    }
    
    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear // Make it transparent
        isOpaque = false
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func draw(_ rect: CGRect) {
        super.draw(rect)
        
        guard let context = UIGraphicsGetCurrentContext() else { return }
        let blackoutRect = CGRect(x: 0, y: 0, width: self.bounds.width, height: self.bounds.height)
        
        var cutout = CGPath(rect: CGRect.zero, transform: nil)
        
        context.setStrokeColor(UIColor.green.cgColor)
        context.setLineWidth(3)
        
        for box in boundingBoxes {
            print("Box: (x = \((box.minX + box.maxX)/2), y = \((box.minY + box.maxY)/2), w = \(box.width), h = \(box.height)), \t\t Bounds: \(self.bounds)")
            // Here, self.bounds is the frame of this view, which is sized to match the preview layer.
            let drawingRect = VNImageRectForNormalizedRect(box, Int(self.bounds.width), Int(self.bounds.height))
            
            let size = max(drawingRect.width, drawingRect.height)
            
            // Flip the Y-coordinate because Vision's origin is bottom-left, and UIKit's is top-left.
            let flippedRect = CGRect(
                x: self.bounds.width - drawingRect.origin.x - drawingRect.width/2 - size/2,
                y: self.bounds.height - drawingRect.origin.y - drawingRect.height,
                width: size, // drawingRect.width,
                height: size // drawingRect.height
            )
            
            //context.stroke(flippedRect)
            let rectPath = CGPath(rect: flippedRect, transform: nil)
            cutout = cutout.union(rectPath)
        }
        
        let blackoutPath = CGPath(rect: blackoutRect, transform: nil).subtracting(cutout)
        context.addPath(blackoutPath)
        context.setFillColor(UIColor.black.cgColor)
        context.setBlendMode(.normal)
        context.drawPath(using: .eoFill)
    }
}

// The UIViewRepresentable now manages a container view that holds both
// the camera preview layer and the drawing view on top.
struct CameraPreview: UIViewRepresentable {
    @ObservedObject var cameraManager: CameraManager
    
    func makeUIView(context: Context) -> UIView {
        let view = UIView(frame: UIScreen.main.bounds)
        
        // Setup the preview layer
        let previewLayer = AVCaptureVideoPreviewLayer()
        previewLayer.videoGravity = .resizeAspectFill
        view.layer.addSublayer(previewLayer)
        
        // Setup the drawing layer
        let drawingView = DrawingView(frame: view.bounds)
        drawingView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(drawingView)
        
        // Store the views in the coordinator to update them later
        context.coordinator.previewLayer = previewLayer
        context.coordinator.drawingView = drawingView
        
        return view
    }
    
    func updateUIView(_ uiView: UIView, context: Context) {
        // Update session if it's newly available
        if context.coordinator.previewLayer?.session == nil, let session = cameraManager.captureSession {
            context.coordinator.previewLayer?.session = session
        }
        
        // Update layer frames on size change
        context.coordinator.previewLayer?.frame = uiView.bounds
        context.coordinator.drawingView?.frame = uiView.bounds
        
        // Pass the latest bounding boxes to the drawing view
        // The drawingView will automatically redraw itself when this property is set.
        context.coordinator.drawingView?.boundingBoxes = cameraManager.boundingBoxes.map { $0.rect }
    }
    
    func makeCoordinator() -> Coordinator {
        Coordinator()
    }
    
    // Coordinator to hold references to our UIKit views
    class Coordinator {
        var previewLayer: AVCaptureVideoPreviewLayer?
        var drawingView: DrawingView?
    }
}

// The ContentView becomes very clean, as all the complex drawing logic
// is now encapsulated in the CameraPreview representable.
struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    
    var body: some View {
        CameraPreview(cameraManager: cameraManager)
            .ignoresSafeArea()
            .onAppear(perform: cameraManager.startSession)
            .onDisappear(perform: cameraManager.stopSession)
    }
}

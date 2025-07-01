//
//  VideoBuffer.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/24/25.
//

import Foundation
@preconcurrency import CoreML
@preconcurrency import CoreVideo
@preconcurrency import Vision

extension ASD {
    final class VideoBuffer: Utils.MLBuffer {
        public let frameSize: CGSize        
        private let cropScale: CGFloat
        private var frameMod6: Int = 0
        
        init(frontPadding: Int = 3, backPadding: Int = 25, cropPadding: CGFloat = 0.40,
             length: Int = 25, frameSize: CGSize = .init(width: 112, height: 112)) {
            self.cropScale = cropPadding
            self.frameSize = frameSize
            super.init(
                chunkShape: [Int(frameSize.width), Int(frameSize.height)],
                defaultChunk: .init(repeating: 110, count: Int(frameSize.width * frameSize.height)),
                length: length,
                frontPadding: frontPadding,
                backPadding: backPadding
            )
        }
        
        public func write(from pixelBuffer: CVPixelBuffer, croppedTo rect: CGRect) {
            self.frameMod6 += 1
            self.frameMod6 %= 6
            if self.frameMod6 == 0 { return }
            
            let cropRect = self.computeCropRect(pixelBuffer: pixelBuffer, rect: rect)
            let _ = self.withUnsafeWritingPointer {
                try? preprocessImage(pixelBuffer: pixelBuffer,
                                     cropTo: cropRect,
                                     resizeTo: self.frameSize,
                                     to: $0.baseAddress!)
            }
        }
        
        @inline(__always)
        private func computeCropRect(pixelBuffer: CVPixelBuffer, rect detectionRect: CGRect) -> CGRect {
            /*
             cs = args.cropScale
             bs = dets['s'][fidx]  # Detection box size
             bsi = int(bs * (1 + 2 * cs))  # Pad videos by this amount
             image = cv2.imread(flist[frame])
             frame = numpy.pad(image, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
             my = dets['y'][fidx] + bsi  # BBox center Y
             mx = dets['x'][fidx] + bsi  # BBox center X
             face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
             vOut.write(cv2.resize(face, (224, 224)))
             
             face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
             face = cv2.resize(face, (224, 224))
             face = face[int(112 - (112 / 2)):int(112 + (112 / 2)), int(112 - (112 / 2)):int(112 + (112 / 2))]
             videoFeature.append(face)
             */
            
            let bufferWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let bufferHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            // Get detection box dimensions and center in pixels.
            let detectionWidth = detectionRect.width * bufferWidth
            let detectionHeight = detectionRect.height * bufferHeight
            let detectionCenterX = detectionRect.midX * bufferWidth
            let detectionCenterY = detectionRect.midY * bufferHeight

            // 'bs' is half the size of the largest dimension of the detection box.
            let bs = max(detectionWidth, detectionHeight) / 2.0
            
            // 'cs' is the crop scale factor.
            let cs = cropScale

            // The Python script calculates an intermediate crop centered at `(mx, my + bs*cs)`.
            // Let's find the center of that intermediate crop.
            let intermediateCropCenterX = detectionCenterX
            let intermediateCropCenterY = detectionCenterY + (bs * cs)
            
            // --- Step 2: Calculate the final crop rect ---
            
            // The Python script resizes the intermediate crop to 224x224 and then takes
            // a 112x112 center crop. This is equivalent to taking a center crop of the
            // intermediate rect that is half its size.
            
            // The side length of the intermediate crop is `2 * bs * (1 + cs)`.
            // The side length of our final crop will be half of that.
            let finalSideLength = bs * (1.0 + cs)
            let finalHalfSide = finalSideLength / 2.0
            
            // Calculate the origin of the final crop rectangle.
            let finalOriginX = intermediateCropCenterX - finalHalfSide
            let finalOriginY = intermediateCropCenterY - finalSideLength
            
            return CGRect(
                x: finalOriginX,
                y: finalOriginY,
                width: finalSideLength,
                height: finalSideLength
            )
        }
    }
}

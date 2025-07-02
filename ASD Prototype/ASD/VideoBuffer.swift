//
//  VideoBuffer.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/24/25.
//

import Foundation
import CoreML
import CoreVideo
import CoreGraphics
import Vision
import Accelerate

extension ASD {
    final class VideoBuffer: Utils.MLBuffer {
        private enum ImageProcessingError: Error {
            case lockFailed
            case unsupportedFormat
            case resizeFailed(vImage_Error)
            case grayscaleFailed(vImage_Error)
            case convertFailed(vImage_Error)
        }
        
        public let frameSize: CGSize
        public private(set) var cropRect: CGRect
        
        private let cropScale: CGFloat
        
        init(frontPadding: Int = 0,
             backPadding: Int = 25,
             cropPadding: CGFloat = asdCropPadding,
             length: Int = asdVideoLength,
             frameSize: CGSize = asdFrameSize)
        {
            self.cropScale = cropPadding
            self.frameSize = frameSize
            self.cropRect = .zero
            super.init(
                chunkShape: [Int(frameSize.width), Int(frameSize.height)],
                defaultChunk: .init(repeating: 110, count: Int(frameSize.width * frameSize.height)),
                length: length,
                frontPadding: frontPadding,
                backPadding: backPadding
            )
        }
        
        public func write(from pixelBuffer: CVPixelBuffer, croppedTo rect: CGRect, skip: Bool = false) {
            self.cropRect = self.computeCropRect(pixelBuffer: pixelBuffer, rect: rect)
            if skip == false {
                let _ = self.withUnsafeWritingPointer {
                    do {
                        try VideoBuffer.preprocessImage(pixelBuffer: pixelBuffer,
                                                        cropTo: cropRect,
                                                        resizeTo: self.frameSize,
                                                        to: $0.baseAddress!)
                    } catch {
                        fatalError(error.localizedDescription)
                    }
                }
            }
        }
        
        @inline(__always)
        private func computeCropRect(pixelBuffer: CVPixelBuffer, rect detectionRect: CGRect) -> CGRect {
            let bufferWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let bufferHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            // Get detection box dimensions and center in pixels.
            let detectionWidth = detectionRect.width * bufferWidth
            let detectionHeight = detectionRect.height * bufferHeight
            let detectionCenterX = detectionRect.midX * bufferWidth
            let detectionCenterY = detectionRect.midY * bufferHeight

            let bs = max(detectionWidth, detectionHeight) / 2.0 // box size
            let cs = self.cropScale
            
            let finalSideLength = bs * (1.0 + cs)
            let finalHalfSide = finalSideLength / 2.0
            
            let intermediateCropCenterX = detectionCenterX
            let intermediateCropCenterY = detectionCenterY - (bs * cs)
            
            let finalOriginX = intermediateCropCenterX - finalHalfSide
            let finalOriginY = intermediateCropCenterY - finalHalfSide
            
            return CGRect(
                x: finalOriginX,
                y: finalOriginY,
                width: finalSideLength,
                height: finalSideLength
            )
        }
        
        /// Crops, resizes, converts to grayscale, and flattens a `CVPixelBuffer` into a
        /// contiguous array of `Float32` values.
        ///
        /// This function creates an optimized pipeline that minimizes intermediate memory
        /// allocations and copies. It does **not** modify the original `CVPixelBuffer`.
        /// The process is as follows:
        /// 1. Crops the source buffer to `cropRect`, padding if necessary, into a temporary buffer.
        /// 2. Resizes the cropped image to `targetSize` into a second temporary buffer.
        /// 3. Converts the resized BGRA image to a single-channel (planar) 8-bit grayscale image.
        /// 4. Converts the 8-bit grayscale image to a 32-bit float grayscale image.
        /// 5. Copies the final flat array of floats to the destination pointer.
        ///
        /// - Parameters:
        ///   - cropRect: The `CGRect` defining the region to crop. Out-of-bounds areas are padded.
        ///   - targetSize: The final `CGSize` for the output data (e.g., 224x224).
        ///   - outputPointer: An `UnsafeMutablePointer<Float32>` to which the final, flattened
        ///                  grayscale data will be written. The pointer must point to a memory
        ///                  region large enough to hold `targetSize.width * targetSize.height` floats.
        /// - Throws: An `ImageProcessingError` if any step in the vImage pipeline fails.
        private static func preprocessImage(
            pixelBuffer: CVPixelBuffer,
            cropTo cropRect: CGRect,
            resizeTo targetSize: CGSize,
            to outputBuffer: UnsafeMutablePointer<Float>
        ) throws {
            // 1) Validate format & lock
            guard CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA else {
                throw ImageProcessingError.unsupportedFormat
            }
            CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
            defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
            guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
                throw ImageProcessingError.lockFailed
            }
            
            // 2) Compute source & crop geometry
            let srcW       = CVPixelBufferGetWidth(pixelBuffer)
            let srcH       = CVPixelBufferGetHeight(pixelBuffer)
            let srcStride  = CVPixelBufferGetBytesPerRow(pixelBuffer)
            let cropX      = Int(cropRect.origin.x)
            let cropY      = srcH - Int(cropRect.maxY)
            let cropW      = Int(cropRect.width)
            let cropH      = Int(cropRect.height)
            let cropStride = cropW * 4
            
            // 3) Allocate & fill cropBuffer with pad color A=255, R=110, G=110, B=110
            let cropData = UnsafeMutableRawPointer.allocate(byteCount: cropH * cropStride,
                                                            alignment: 64)
            defer { cropData.deallocate() }
            var cropBuffer = vImage_Buffer(data:       cropData,
                                           height:     vImagePixelCount(cropH),
                                           width:      vImagePixelCount(cropW),
                                           rowBytes:   cropStride)
            
            // Pad color in BGRA memory layout → we’ll permute later anyway
            var padColor: [UInt8] = [110, 110, 110, 255]
            let fillErr = vImageBufferFill_ARGB8888(&cropBuffer,
                                                    &padColor,
                                                    vImage_Flags(kvImageNoFlags))
            guard fillErr == kvImageNoError else {
                throw ImageProcessingError.resizeFailed(fillErr)
            }
            
            // 4) Copy overlapping region from source into the right offset inside cropBuffer
            let xSrcStart = max(0, cropX)
            let ySrcStart = max(0, cropY)
            let xSrcEnd   = min(srcW, cropX + cropW)
            let ySrcEnd   = min(srcH, cropY + cropH)
            
            if xSrcEnd > xSrcStart && ySrcEnd > ySrcStart {
                let copyW = (xSrcEnd - xSrcStart) * 4
                let copyH = ySrcEnd - ySrcStart
                let destX = (xSrcStart - cropX) * 4
                let destY =  ySrcStart - cropY
                
                for row in 0..<copyH {
                    let srcRowPtr = baseAddress.advanced(
                        by: (ySrcStart + row) * srcStride + xSrcStart * 4
                    )
                    let dstRowPtr = cropData.advanced(
                        by: (destY + row) * cropStride + destX
                    )
                    memcpy(dstRowPtr, srcRowPtr, copyW)
                }
            }
            
            // 5) Now scale the padded cropBuffer → scaledBuffer
            let destW      = Int(targetSize.width)
            let destH      = Int(targetSize.height)
            let destStride = destW * 4
            let scaledData = UnsafeMutableRawPointer.allocate(byteCount: destH * destStride,
                                                              alignment: 64)
            defer { scaledData.deallocate() }
            var scaledBuffer = vImage_Buffer(data:     scaledData,
                                             height:   vImagePixelCount(destH),
                                             width:    vImagePixelCount(destW),
                                             rowBytes: destStride)
            var err = vImageScale_ARGB8888(&cropBuffer,
                                           &scaledBuffer,
                                           nil,
                                           vImage_Flags(kvImageHighQualityResampling))
            guard err == kvImageNoError else {
                throw ImageProcessingError.resizeFailed(err)
            }
            
            // 6) Permute BGRA→ARGB for the luma matrix multiply
            let permuteMap: [UInt8] = [3, 2, 1, 0]
            let permData = UnsafeMutableRawPointer.allocate(byteCount: destH * destStride,
                                                            alignment: 64)
            defer { permData.deallocate() }
            var permBuffer = vImage_Buffer(data:     permData,
                                           height:   vImagePixelCount(destH),
                                           width:    vImagePixelCount(destW),
                                           rowBytes: destStride)
            err = vImagePermuteChannels_ARGB8888(&scaledBuffer,
                                                 &permBuffer,
                                                 permuteMap,
                                                 vImage_Flags(kvImageNoFlags))
            guard err == kvImageNoError else {
                throw ImageProcessingError.resizeFailed(err)
            }
            
            // 7) Convert ARGB8888 → 8-bit luma
            let gray8Stride = destW
            let gray8Data = UnsafeMutableRawPointer.allocate(byteCount: destH * gray8Stride,
                                                             alignment: 1)
            defer { gray8Data.deallocate() }
            var gray8Buffer = vImage_Buffer(data:     gray8Data,
                                            height:   vImagePixelCount(destH),
                                            width:    vImagePixelCount(destW),
                                            rowBytes: gray8Stride)
            
            let divisor: Int32 = 0x1000
            let rCoef = Int16(0.299 * Float(divisor))
            let gCoef = Int16(0.587 * Float(divisor))
            let bCoef = Int16(0.114 * Float(divisor))
            var matrix: [Int16] = [ 0, rCoef, gCoef, bCoef ]
            var preBias = [Int16](repeating: 0, count: 4)
            let postBias: Int32 = 0
            
            err = vImageMatrixMultiply_ARGB8888ToPlanar8(&permBuffer,
                                                         &gray8Buffer,
                                                         &matrix,
                                                         divisor,
                                                         &preBias,
                                                         postBias,
                                                         vImage_Flags(kvImageNoFlags))
            guard err == kvImageNoError else {
                throw ImageProcessingError.grayscaleFailed(err)
            }
            
            // 8) Convert Planar8 → PlanarF (Float32) into user’s outputBuffer
            let outStride = destW * MemoryLayout<Float>.stride
            var grayFBuffer = vImage_Buffer(data:     outputBuffer,
                                            height:   vImagePixelCount(destH),
                                            width:    vImagePixelCount(destW),
                                            rowBytes: outStride)
            err = vImageConvert_Planar8toPlanarF(&gray8Buffer,
                                                 &grayFBuffer,
                                                 255.0,  // scale
                                                 0.0,    // bias
                                                 vImage_Flags(kvImageNoFlags))
            guard err == kvImageNoError else {
                throw ImageProcessingError.convertFailed(err)
            }
        }
    }
}

import Foundation
import Accelerate
import CoreVideo
import UIKit // For CGRect extensions, can be replaced with CoreGraphics

// Define a custom error type for more specific error handling.
enum ImageProcessingError: Error {
    case allocationFailed(String)
    case lockBaseAddressFailed(String)
    case operationFailed(String, vImage_Error)
}

extension CVPixelBuffer {
    
    /// Crops a `CVPixelBuffer` to a specified square `CGRect`, pads if the crop rect is out of bounds,
    /// and resizes it to a target square size.
    ///
    /// This function is optimized for performance by using the Accelerate framework's vImage library.
    ///
    /// - Parameters:
    ///   - cropRect: A `CGRect` defining the square region to crop from the source buffer.
    ///               If this rect extends beyond the buffer's bounds, the missing area will be padded.
    ///   - targetSize: The square `CGSize` for the final output buffer (e.g., 224x224).
    /// - Returns: A new `CVPixelBuffer` containing the cropped, padded, and resized image.
    /// - Throws: An `ImageProcessingError` if any step in the vImage pipeline fails.
    func croppedAndResized(to targetSize: CGSize, from cropRect: CGRect) throws -> CVPixelBuffer {
        // --- 1. Validate Input & Get Source Buffer Properties ---
        let sourceWidth = CVPixelBufferGetWidth(self)
        let sourceHeight = CVPixelBufferGetHeight(self)
        let pixelFormat = CVPixelBufferGetPixelFormatType(self)
        
        guard pixelFormat == kCVPixelFormatType_32BGRA || pixelFormat == kCVPixelFormatType_32ARGB else {
            // vImage works best with 32-bit pixel formats.
            throw ImageProcessingError.operationFailed("Unsupported pixel format: \(pixelFormat)", kvImageInvalidParameter)
        }
        
        // Lock the source buffer's base address for reading.
        CVPixelBufferLockBaseAddress(self, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(self, .readOnly) }

        guard let sourceBaseAddress = CVPixelBufferGetBaseAddress(self) else {
            throw ImageProcessingError.lockBaseAddressFailed("Could not get source CVPixelBuffer base address.")
        }
        
        // Create a vImage_Buffer for the source image.
        var sourceBuffer = vImage_Buffer(
            data: sourceBaseAddress,
            height: vImagePixelCount(sourceHeight),
            width: vImagePixelCount(sourceWidth),
            rowBytes: CVPixelBufferGetBytesPerRow(self)
        )
        
        // --- 2. Create a Temporary Buffer for the Cropped & Padded Image ---
        
        // The temporary buffer will have the dimensions of the crop rectangle, truncated to integers.
        let cropWidth = Int(cropRect.width)
        let cropHeight = Int(cropRect.height)

        // Allocate a temporary buffer to hold the cropped (and potentially padded) image.
        var cropAndPadBuffer: vImage_Buffer
        do {
            cropAndPadBuffer = try vImage_Buffer(width: cropWidth, height: cropHeight, bitsPerPixel: 32)
        } catch {
            throw ImageProcessingError.allocationFailed("Failed to allocate temporary crop buffer.")
        }
        defer { free(cropAndPadBuffer.data) } // Ensure this buffer is deallocated.

        // --- 3. Pad the Temporary Buffer ---
        
        // Define the padding color [B, G, R, A] for BGRA format. (110, 110, 110)
        let paddingColor: [UInt8] = [110, 110, 110, 255]

        // Fill the entire temporary buffer with the padding color.
        // This handles all out-of-bounds areas in a single, efficient operation.
        let fillError = vImageBufferFill_ARGB8888(&cropAndPadBuffer, paddingColor, 0)
        guard fillError == kvImageNoError else {
            throw ImageProcessingError.operationFailed("Failed to fill padding buffer", fillError)
        }
        
        // --- 4. Copy Valid Pixels from Source to Temporary Buffer (CRASH FIX APPLIED) ---
        
        // Calculate the rectangle of valid pixels that intersects the source buffer.
        let sourceBounds = CGRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight)
        let validSourceRect = sourceBounds.intersection(cropRect)
        
        if !validSourceRect.isNull {
            // We must use integer coordinates for buffer manipulation.
            // The crash occurs when floating-point inaccuracies cause the calculated copy size
            // to be one pixel larger than the destination buffer's allocated size.
            
            // Calculate the integer coordinates for the top-left corner of the copy
            // destinations and sources. We use floor to be conservative.
            let destX = Int(floor(validSourceRect.minX - cropRect.minX))
            let destY = Int(floor(validSourceRect.minY - cropRect.minY))
            
            let sourceX = Int(floor(validSourceRect.minX))
            let sourceY = Int(floor(validSourceRect.minY))
            
            // CRITICAL FIX: Calculate the width and height of the copy.
            // This size must fit in BOTH the source and destination buffers from their respective
            // start points. We clamp the copy dimensions to prevent any buffer overruns.
            let copyWidth = min(
                Int(floor(validSourceRect.width)), // How much valid data there is
                cropWidth - destX                  // How much space is available in the destination
            )
            let copyHeight = min(
                Int(floor(validSourceRect.height)), // How much valid data there is
                cropHeight - destY                  // How much space is available in the destination
            )

            // Ensure we don't try to copy a negative-sized block.
            guard copyWidth > 0 && copyHeight > 0 else {
                // If there's nothing to copy, we can proceed directly to resizing the padded buffer.
                // This 'guard' is a safeguard; normal logic should proceed.
                return try resizePaddedBuffer(buffer: cropAndPadBuffer, targetSize: targetSize, pixelFormat: pixelFormat)
            }

            // Create a pointer to the destination region in the temporary buffer.
            let destData = cropAndPadBuffer.data.advanced(by: destY * cropAndPadBuffer.rowBytes + destX * 4)
            var destCropBuffer = vImage_Buffer(
                data: destData,
                height: vImagePixelCount(copyHeight), // Use SAFE clamped height
                width: vImagePixelCount(copyWidth),   // Use SAFE clamped width
                rowBytes: cropAndPadBuffer.rowBytes
            )
            
            // Create a pointer to the source region in the source buffer.
            let sourceData = sourceBuffer.data.advanced(by: sourceY * sourceBuffer.rowBytes + sourceX * 4)
            var sourceCropBuffer = vImage_Buffer(
                data: sourceData,
                height: vImagePixelCount(copyHeight), // Use SAFE clamped height
                width: vImagePixelCount(copyWidth),   // Use SAFE clamped width
                rowBytes: sourceBuffer.rowBytes
            )

            // Copy the valid pixel data. vImageScale with no flags is equivalent to a fast copy.
            let copyError = vImageScale_ARGB8888(&sourceCropBuffer, &destCropBuffer, nil, vImage_Flags(kvImageNoFlags))
            guard copyError == kvImageNoError else {
                throw ImageProcessingError.operationFailed("Failed to copy valid pixels to padded buffer", copyError)
            }
        }
        
        // --- 5. Create Destination Buffer and Perform Resize ---
        return try resizePaddedBuffer(buffer: cropAndPadBuffer, targetSize: targetSize, pixelFormat: pixelFormat)
    }
    
    /// Helper function to resize a vImage_Buffer to a CVPixelBuffer.
    private func resizePaddedBuffer(buffer: vImage_Buffer, targetSize: CGSize, pixelFormat: OSType) throws -> CVPixelBuffer {
        // Create the final destination pixel buffer.
        var destPixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            Int(targetSize.width),
            Int(targetSize.height),
            pixelFormat,
            nil,
            &destPixelBuffer
        )
        guard status == kCVReturnSuccess, let finalPixelBuffer = destPixelBuffer else {
            throw ImageProcessingError.allocationFailed("Failed to create destination CVPixelBuffer.")
        }

        // Lock its address for writing.
        CVPixelBufferLockBaseAddress(finalPixelBuffer, CVPixelBufferLockFlags(rawValue: 0))
        defer { CVPixelBufferUnlockBaseAddress(finalPixelBuffer, CVPixelBufferLockFlags(rawValue: 0)) }
        
        guard let destBaseAddress = CVPixelBufferGetBaseAddress(finalPixelBuffer) else {
            throw ImageProcessingError.lockBaseAddressFailed("Could not get destination CVPixelBuffer base address.")
        }
        
        // Create a mutable copy of the input buffer to pass to the resize function.
        var mutableBuffer = buffer
        
        // Create a vImage_Buffer for the destination.
        var destBuffer = vImage_Buffer(
            data: destBaseAddress,
            height: vImagePixelCount(targetSize.height),
            width: vImagePixelCount(targetSize.width),
            rowBytes: CVPixelBufferGetBytesPerRow(finalPixelBuffer)
        )
        
        // Perform the high-quality resize from the temporary (cropped and padded) buffer
        // to the final destination buffer.
        let resizeError = vImageScale_ARGB8888(&mutableBuffer, &destBuffer, nil, vImage_Flags(kvImageHighQualityResampling))
        guard resizeError == kvImageNoError else {
            throw ImageProcessingError.operationFailed("Failed to resize image", resizeError)
        }
        
        return finalPixelBuffer
    }
}

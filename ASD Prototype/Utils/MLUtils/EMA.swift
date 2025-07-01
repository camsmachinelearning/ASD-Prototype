import CoreML
import Accelerate

extension Utils.ML {
    static func updateEMA(ema: MLMultiArray, with newArray: MLMultiArray, alpha: Float) {
        precondition(ema.count == newArray.count, "Shape mismatch")
        precondition(ema.dataType == .float32 && newArray.dataType == .float32, "Only Float32 supported")
        
        let count = ema.count
        let emaPtr = UnsafeMutableBufferPointer<Float>(
            start: UnsafeMutablePointer<Float>(OpaquePointer(ema.dataPointer)),
            count: count
        )
        
        let newPtr = UnsafeBufferPointer<Float>(
            start: UnsafePointer<Float>(OpaquePointer(newArray.dataPointer)),
            count: count
        )
        
        // Compute: ema ← α * new + (1 - α) * ema
        var alpha = alpha
        var oneMinusAlpha = 1 - alpha
        
        // tmp = alpha * new
        var tmp = [Float](repeating: 0, count: count)
        vDSP_vsmul(newPtr.baseAddress!, 1, &alpha, &tmp, 1, vDSP_Length(count))
        
        // ema = (1 - alpha) * ema
        vDSP_vsmul(emaPtr.baseAddress!, 1, &oneMinusAlpha, emaPtr.baseAddress!, 1, vDSP_Length(count))
        
        // ema += tmp
        vDSP_vadd(emaPtr.baseAddress!, 1, &tmp, 1, emaPtr.baseAddress!, 1, vDSP_Length(count))
    }
}

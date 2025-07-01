import CoreML
import Accelerate

extension Utils.ML {
    static func cosineDistance(a: MLMultiArray, b: MLMultiArray) -> Float {
        // 1. Quick shape check
        precondition(a.count == b.count)
        precondition(a.dataType == .float32 && b.dataType == .float32)
        
        // 2. Bind the raw pointers
        let aPtr = a.dataPointer.bindMemory(to: Float.self, capacity: a.count)
        let bPtr = b.dataPointer.bindMemory(to: Float.self, capacity: b.count)
        let n = vDSP_Length(a.count)
        
        // 3. Dot product
        var dot: Float = 0
        vDSP_dotpr(aPtr, 1, bPtr, 1, &dot, n)
        
        // 4. Squared norms
        var normASq: Float = 0
        var normBSq: Float = 0
        vDSP_svesq(aPtr, 1, &normASq, n)
        vDSP_svesq(bPtr, 1, &normBSq, n)
        
        let normANormB = sqrt(normASq * normBSq)
        
        // 6. Final cosine similarity
        return 1.0 - dot / normANormB
    }
}

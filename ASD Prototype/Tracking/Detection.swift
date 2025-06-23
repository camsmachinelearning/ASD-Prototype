import Foundation
import CoreVideo
import CoreGraphics
import CoreML


extension Tracking {
    final class Detection:
        Identifiable,
        Hashable,
        Equatable
    {
        let id = UUID()
        let rect: CGRect
        let confidence: Float
        
        var buffer: CVPixelBuffer?
        var embedding: MLMultiArray?
        
        init (rect: CGRect, confidence: Float) {
            self.rect = rect
            self.confidence = confidence
        }
        
        static func == (lhs: Detection, rhs: Detection) -> Bool {
            return lhs.id == rhs.id
        }
        
        func hash(into hasher: inout Hasher) {
            hasher.combine(id)
        }
    }
}

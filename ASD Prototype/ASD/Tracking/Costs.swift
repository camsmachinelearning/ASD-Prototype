//
//  Costs.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/24/25.
//

import Foundation

extension ASD.Tracking {
    final class Costs {
        var iou: Float
        var appearance: Float
        var total: Float
        
        var hasIoU: Bool { self.iou != Float.infinity }
        var hasAppearance: Bool { self.appearance != Float.infinity }
        var hasTotal: Bool { self.total != Float.infinity }
        
        var string : String {
            let totalString: String = self.hasTotal ? "Cost = \(self.total)" : "Cost:"
            let iouString: String = self.hasIoU ? "\n\tIoU: \(self.iou)" : "\n"
            let appearanceString: String = self.hasAppearance ? "\n\tAppearance: \(self.appearance)" : "\n"
            return "\(totalString)\(iouString)\(appearanceString)"
        }
        
        init(iou: Float = Float.infinity, appearance: Float = Float.infinity, total: Float = Float.infinity) {
            self.iou = iou
            self.appearance = appearance
            self.total = total
        }
    }
}

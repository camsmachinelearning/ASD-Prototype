//
//  LAP.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/17/25.
//

import Foundation
import RLAP

extension Utils {
    @usableFromInline
    @discardableResult
    static func solveRLAP(
        dims: (Int, Int),
        cost: [Float],
        rows: inout [Int],
        cols: inout [Int],
        maximize: Bool = false
    ) -> Int {
        var rowSol = [Int32](repeating: 0, count: dims.0)
        var colSol = [Int32](repeating: 0, count: dims.1)
        
        // Some very safe code the definitely won't cause segfaults :)
        let returnCode = solve_rlapf(
            Int32(dims.0),
            Int32(dims.1),
            cost.withUnsafeBufferPointer { $0.baseAddress! },
            maximize,
            rowSol.withUnsafeMutableBufferPointer { $0.baseAddress! },
            colSol.withUnsafeMutableBufferPointer { $0.baseAddress! }
        )
        
        rows = rowSol.map(Int.init)
        cols = colSol.map(Int.init)
        
        return Int(returnCode)
    }
    
    @usableFromInline
    @discardableResult
    static func solveRLAP(
        dims: (Int, Int),
        cost: [Double],
        rows: inout [Int],
        cols: inout [Int],
        maximize: Bool = false
    ) -> Int {
        var rowSol = [Int32](repeating: 0, count: dims.0)
        var colSol = [Int32](repeating: 0, count: dims.1)
        
        // Some very safe code the definitely won't cause segfaults :)
        let returnCode = solve_rlap(
            Int32(dims.0),
            Int32(dims.1),
            cost.withUnsafeBufferPointer { $0.baseAddress! },
            maximize,
            rowSol.withUnsafeMutableBufferPointer { $0.baseAddress! },
            colSol.withUnsafeMutableBufferPointer { $0.baseAddress! }
        )
        
        rows = rowSol.map(Int.init)
        cols = colSol.map(Int.init)
        
        return Int(returnCode)
    }
}

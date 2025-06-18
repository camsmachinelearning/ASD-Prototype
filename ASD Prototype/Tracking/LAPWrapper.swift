//
//  LAP.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/17/25.
//

import Foundation
import LAP

@usableFromInline
@discardableResult
func solveLap(
    inputDims: Int,
    outputDims: Int,
    cost: [Float],
    rowSolution: inout [Int32],
    colSolution: inout [Int32]
) -> Float {
    if (rowSolution.count != inputDims) {
        rowSolution = Array(repeating: 0, count: inputDims)
    }
    if colSolution.count != outputDims {
        colSolution = Array(repeating: 0, count: outputDims)
    }
    
    return cost.withUnsafeBufferPointer { costPtr in
        rowSolution.withUnsafeMutableBufferPointer { rowPtr in
            colSolution.withUnsafeMutableBufferPointer { colPtr in
                solve_lapf(
                    Int32(inputDims),
                    Int32(outputDims),
                    costPtr.baseAddress!,
                    rowPtr.baseAddress!,
                    colPtr.baseAddress!
                )
            }
        }
    }
}

@usableFromInline
@discardableResult 
func solveLap(
    inputDims: Int,
    outputDims: Int,
    cost: [Double],
    rowSolution: inout [Int32],
    colSolution: inout [Int32]
) -> Double {
    if (rowSolution.count != inputDims) {
        rowSolution = Array(repeating: 0, count: inputDims)
    }
    if colSolution.count != outputDims {
        colSolution = Array(repeating: 0, count: outputDims)
    }
    
    return cost.withUnsafeBufferPointer { costPtr in
        rowSolution.withUnsafeMutableBufferPointer { rowPtr in
            colSolution.withUnsafeMutableBufferPointer { colPtr in
                solve_lap(
                    Int32(inputDims),
                    Int32(outputDims),
                    costPtr.baseAddress!,
                    rowPtr.baseAddress!,
                    colPtr.baseAddress!
                )
            }
        }
    }
}



//
//  mod.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/29/25.
//

import Foundation

extension Utils {
    @inline(__always)
    static func mod(_ a: Int, _ b: Int) -> Int {
        let floorAOverB = (a - b + 1) / b
        return a - floorAOverB * b
    }
}

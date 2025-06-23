//
//  TrackData.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/21/25.
//

import Foundation

extension Tracking {
    struct Face:
        Sendable,
        Identifiable,
        Hashable,
        Equatable
    {
        let id: UUID
        let rect: CGRect
        let costString: String
        let status: Track.Status
        let misses: Int
        
        var string: String { "ID: \(id.uuidString)\n\(self.costString)" }
        
        init(track: Track) {
            self.id = track.id
            self.rect = track.rect
            self.status = track.status
            self.costString = "\(track.costs.string)\nAppearance (Average): \(track.averageAppearanceCost)"
            self.misses = -track.hits
        }
    }
}

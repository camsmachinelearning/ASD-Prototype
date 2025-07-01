//
//  TrackData.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/21/25.
//

import Foundation
import CoreVideo
import CoreGraphics
import Vision


extension ASD {
    struct SpeakerData:
        Sendable,
        Identifiable,
        Hashable,
        Equatable
    {
        let id: UUID
        let rect: CGRect
        let costString: String
        let status: Tracking.Track.Status
        let misses: Int
        let score: Float
        
        var string: String { "ID: \(id.uuidString)\n\(self.costString)" }
        
        init(track: Tracking.Tracker.SendableTrack, score: Float, rect: CGRect? = nil) {
            self.id = track.id
            self.rect = rect ?? track.rect
            self.status = track.status
            self.costString = track.costString
            self.misses = track.misses
            self.score = score
        }
        
        static func == (lhs: SpeakerData, rhs: SpeakerData) -> Bool {
            return lhs.id == rhs.id
        }
        
        nonisolated public func hash(into hasher: inout Hasher) {
            hasher.combine(id)
        }
    }
}

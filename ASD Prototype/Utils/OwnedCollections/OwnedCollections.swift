//
//  OwnedContainers.swift
//  ASD Prototype
//
//  Created by Benjamin Lee on 6/22/25.
//

import Foundation
import OrderedCollections

class OwnedCollection<T> where T: Sequence {
    public var collection: T
    
    @usableFromInline
    @inline(__always)
    init(_ collection: T) {
        self.collection = collection
    }
}

typealias OwnedOrderedSet<E> = OwnedCollection<OrderedSet<E>> where E: Hashable
typealias OwnedSet<E> = OwnedCollection<Set<E>> where E: Hashable
typealias OwnedOrderedDictionary<K, V> = OwnedCollection<OrderedDictionary<K, V>> where K: Hashable
typealias OwnedDictionary<K, V> = OwnedCollection<Dictionary<K, V>> where K: Hashable
typealias OwnedArray<E> = OwnedCollection<Array<E>>

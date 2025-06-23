import Foundation

/// A collection that supports O(1) append, random access, removal, sorting,
/// and iteration, mirroring Swift's Array protocols.
struct IndexedSet<Element: Identifiable>:
    RandomAccessCollection,
    MutableCollection,
    RangeReplaceableCollection,
    ExpressibleByArrayLiteral
{
    // MARK: - Underlying Storage
    private(set) var elements: [Element] = []
    private var indexByID: [Element.ID: Int] = [:]
    
    public var count: Int { elements.count }
    
    public var capacity: Int { elements.capacity }
    
    mutating func reserveCapacity(_ newCapacity: Int) {
        elements.reserveCapacity(newCapacity)
        indexByID.reserveCapacity(newCapacity)
    }

    // MARK: - RangeReplaceableCollection
    init() {
        // empty initializer
    }

    init(arrayLiteral elements: Element...) {
        self.init()
        for e in elements { append(e) }
    }

    mutating func replaceSubrange<C: Collection>(_ subrange: Range<Int>, with newElements: C)
        where C.Element == Element
    {
        // Remove old IDs
        for idx in subrange {
            indexByID.removeValue(forKey: elements[idx].id)
        }
        // Replace range
        elements.replaceSubrange(subrange, with: newElements)
        // Rebuild indices map
        for (i, e) in elements.enumerated() {
            indexByID[e.id] = i
        }
    }

    // MARK: - Collection
    typealias Index = Int

    var startIndex: Int { elements.startIndex }
    var endIndex: Int   { elements.endIndex }

    func index(after i: Int) -> Int {
        elements.index(after: i)
    }

    // MARK: - BidirectionalCollection
    func index(before i: Int) -> Int {
        elements.index(before: i)
    }

    // MARK: - MutableCollection
    subscript(position: Int) -> Element {
        get { elements[position] }
        set {
            let oldID = elements[position].id
            indexByID.removeValue(forKey: oldID)
            elements[position] = newValue
            indexByID[newValue.id] = position
        }
    }

    // MARK: - Custom Operations

    /// Append an element in O(1).
    mutating func append(_ e: Element) {
        indexByID[e.id] = elements.count
        elements.append(e)
    }

    /// Find the index for a given ID in O(1).
    func index(ofID id: Element.ID) -> Int? {
        indexByID[id]
    }

    /// Remove an element by ID in O(1).
    @discardableResult
    mutating func remove(id: Element.ID) -> Element? {
        guard let idx = indexByID[id] else { return nil }
        let lastIdx = elements.count - 1
        elements.swapAt(idx, lastIdx)
        let moved = elements[idx]
        indexByID[moved.id] = idx
        let removed = elements.removeLast()
        indexByID.removeValue(forKey: removed.id)
        return removed
    }

    /// Remove and return element at the given index in O(1).
    @discardableResult
    mutating func remove(at index: Int) -> Element? {
        guard elements.indices.contains(index) else { return nil }
        return remove(id: elements[index].id)
    }

    /// Sort elements and rebuild index map in O(n log n).
    mutating func sort(by areInIncreasingOrder: (Element, Element) -> Bool) {
        elements.sort(by: areInIncreasingOrder)
        for (i, e) in elements.enumerated() {
            indexByID[e.id] = i
        }
    }
}



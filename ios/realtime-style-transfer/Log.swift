import Foundation

enum Log {
    static func bench<T>(_ fileID: String = #fileID, _ function: String = #function, _ f: () throws -> T) rethrows -> T {
        try bench("\(fileID).\(function)", f)
    }

    static func bench<T>(_ description: String, _ f: () throws -> T) rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try f()
        let duration = CFAbsoluteTimeGetCurrent() - start
        print("\(description) took \(duration) seconds.")
        return result
    }

    static func bench<T>(_ fileID: String = #fileID, _ function: String = #function, _ f: () async throws -> T) async rethrows -> T {
        try await bench("\(fileID).\(function)", f)
    }

    static func bench<T>(_ description: String, _ f: () async throws -> T) async rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await f()
        let duration = CFAbsoluteTimeGetCurrent() - start
        print("\(description) took \(duration) seconds.")
        return result
    }
}

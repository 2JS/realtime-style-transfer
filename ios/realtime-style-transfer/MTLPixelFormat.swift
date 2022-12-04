import Metal

extension MTLPixelFormat {
    var channels: Int {
        return 4
    }

    var bytesPerPixel: Int {
        switch self {
            case .rgba32Float:
                return 4 * MemoryLayout<Float32>.size
            case .bgra8Unorm, .rgba8Unorm:
                return 4 * MemoryLayout<UInt8>.size
            default: fatalError()
        }
    }

    var bytesPerChannel: Int {
        switch self {
            case .rgba32Float:
                return MemoryLayout<Float32>.size
            case .bgra8Unorm, .rgba8Unorm:
                return MemoryLayout<UInt8>.size
            default: fatalError()
        }
    }
}

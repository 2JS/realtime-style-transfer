import CoreImage
import CoreML
import Metal
import MetalPerformanceShaders

class GPU {
    static let device = MTLCreateSystemDefaultDevice()!
    static let queue = device.makeCommandQueue()!
    static let ciContext = CIContext(mtlCommandQueue: queue)
}

extension MTLTexture {
    static func new(
        height: Int,
        width: Int,
        pixelFormat: MTLPixelFormat,
        usage: MTLTextureUsage,
        resourceOptions: MTLResourceOptions = .storageModeShared,
        on device: MTLDevice = GPU.device
    ) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor()
        descriptor.height = height
        descriptor.width = width
        descriptor.pixelFormat = pixelFormat
        descriptor.usage = usage
        descriptor.resourceOptions = resourceOptions

        return device.makeTexture(descriptor: descriptor)
    }

    static func new_like(
        _ texture: MTLTexture,
        height: Int? = nil,
        width: Int? = nil,
        pixelFormat: MTLPixelFormat? = nil,
        usage: MTLTextureUsage? = nil,
        resourceOptions: MTLResourceOptions? = nil,
        on device: MTLDevice = GPU.device
    ) -> MTLTexture? {
        let descriptor = MTLTextureDescriptor()
        descriptor.height = height ?? texture.height
        descriptor.width = width ?? texture.width
        descriptor.pixelFormat = pixelFormat ?? texture.pixelFormat
        descriptor.usage = usage ?? texture.usage
        descriptor.resourceOptions = resourceOptions ?? texture.resourceOptions

        return device.makeTexture(descriptor: descriptor)
    }

    func converted(pixelFormat: MTLPixelFormat) -> MTLTexture? {
        guard let colorspace = CGColorSpace(name: CGColorSpace.linearSRGB),
              let commandBuffer = GPU.queue.makeCommandBuffer(),
              let destTexture = Self.new_like(self, pixelFormat: pixelFormat, usage: [self.usage, .shaderRead, .shaderWrite])
        else {
            return nil
        }

        let conversionInfo = CGColorConversionInfo(src: colorspace, dst: colorspace)

        let conversion = MPSImageConversion(
            device: GPU.device,
            srcAlpha: .alphaIsOne,
            destAlpha: .alphaIsOne,
            backgroundColor: nil,
            conversionInfo: conversionInfo
        )

        conversion.encode(
            commandBuffer: commandBuffer,
            sourceTexture: self,
            destinationTexture: destTexture
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        return destTexture
    }

    func convert(into: MTLTexture) throws {
        guard let colorspace = CGColorSpace(name: CGColorSpace.linearSRGB),
              let commandBuffer = GPU.queue.makeCommandBuffer()
        else {
            throw GPUError.generic
        }

        let conversionInfo = CGColorConversionInfo(src: colorspace, dst: colorspace)

        let conversion = MPSImageConversion(
            device: GPU.device,
            srcAlpha: .alphaIsOne,
            destAlpha: .alphaIsOne,
            backgroundColor: nil,
            conversionInfo: conversionInfo
        )

        conversion.encode(
            commandBuffer: commandBuffer,
            sourceTexture: self,
            destinationTexture: into
        )

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}

enum GPUError: Swift.Error {
    case generic
}

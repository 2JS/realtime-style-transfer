import AVFoundation
import CoreML
import MetalKit
import MetalPerformanceShaders
import UIKit


let width = 640
let height = 480

class Processor {
    static let shared = Processor()

    private let encoder = try! adain_vgg(configuration: MLModelConfiguration().then {
        $0.computeUnits = .cpuAndGPU
//        $0.allowLowPrecisionAccumulationOnGPU = true
    })

    private let decoder = try! adain_dec(configuration: MLModelConfiguration().then {
        $0.computeUnits = .cpuAndGPU
//        $0.allowLowPrecisionAccumulationOnGPU = true
    })

    private let loader = MTKTextureLoader(device: GPU.device)

    private var modelInputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var modelOutputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var outputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .bgra8Unorm, usage: [.shaderRead, .shaderWrite])!

    var styleInputBuffer = TextureBuffer(device: GPU.device, height: 640, width: 480, pixelFormat: .rgba32Float, usage: [.shaderRead, .shaderWrite])!
    private var styleArray: MLMultiArray?

    var textureCache: CVMetalTextureCache!

    init() {
        CVMetalTextureCacheCreate(kCFAllocatorDefault, nil, GPU.device, nil, &textureCache)
    }

    deinit {
        CVMetalTextureCacheFlush(textureCache, 0)
    }

    func process(sampleBuffer: CMSampleBuffer) -> CIImage? {
        guard //let mtlTexture = getTexture(from: sampleBuffer),
//              (try? mtlTexture.convert(into: modelInputBuffer.texture)) != nil,
//              let input = modelInputBuffer.mlmultiarray,
//              let input = modelInputBuffer.texture.pixelBuffer,
              let input = CMSampleBufferGetImageBuffer(sampleBuffer),
              let latent = encode(input),
              let output = decode(content: latent, style: styleArray)
        else {
            return nil
        }

//        modelOutputBuffer.mlmultiarray = output

//        guard (try? modelOutputBuffer.texture.convert(into: outputBuffer.texture)) != nil
//        else {
//            return nil
//        }
//
        let orientation: CGImagePropertyOrientation
        if ProcessInfo().isiOSAppOnMac {
            orientation = .downMirrored
        } else {
            orientation = .down
        }
//        print(CVPixelBufferGetPixelFormatType(output))
//        print(kCVPixelFormatType_32BGRA, kCVPixelFormatType_32ARGB)
        return CIImage(cvPixelBuffer: output).oriented(orientation)
//        return CIImage(mtlTexture: outputBuffer.texture)?.oriented(orientation)
    }

    func getTexture(from sampleBuffer: CMSampleBuffer) -> MTLTexture? {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer)
        else {
            return nil
        }

        let colorFormat: MTLPixelFormat = .bgra8Unorm
        let width = CVPixelBufferGetWidth(imageBuffer)
        let height = CVPixelBufferGetHeight(imageBuffer)

        var texture: CVMetalTexture!
        let status = CVMetalTextureCacheCreateTextureFromImage(
            kCFAllocatorDefault,
            self.textureCache,
            imageBuffer,
            nil,
            colorFormat,
            width,
            height,
            0,
            &texture
        )

        if status != kCVReturnSuccess {
            return nil
        }

        let mtlTexture = CVMetalTextureGetTexture(texture)!

        return mtlTexture
    }

    func encode(style: UIImage) throws {
        guard let texture = (try? texture(cgImageOf: style)) ?? (try? texture(pngOf: style))
        else {
            throw GPUError.generic
        }

        try texture.convert(into: styleInputBuffer.texture)

        guard //let pixelBuffer = styleInputBuffer.texture.pixelBuffer,
              let cgImage = style.cgImage,
              let style = try? encoder.prediction(input: adain_vggInput(xWith: cgImage)).var_147
        else {
            throw GPUError.generic
        }

        self.styleArray = style
    }

    func discardStyle() {
        self.styleArray = nil
    }

    func encode(_ input: CVPixelBuffer) -> MLMultiArray? {
//        try? encoder.prediction(input: adain_vggInput(xWith: input)).var_160
        let start = CFAbsoluteTimeGetCurrent()
        let result = try? encoder.prediction(x: input).var_147

        let duration = CFAbsoluteTimeGetCurrent() - start
        print("encoder", duration)

        return result
    }

    func decode(content: MLMultiArray, style: MLMultiArray? = nil) -> CVPixelBuffer? {
        let start = CFAbsoluteTimeGetCurrent()
        let result = try? decoder.prediction(content: content, style: style ?? content).y

        let duration = CFAbsoluteTimeGetCurrent() - start
        print("decoder", duration)

        return result
    }

    func texture(cgImageOf image: UIImage) throws -> MTLTexture {
        guard let cgImage = image.cgImage
        else {
            throw GPUError.generic
        }

        let bytesPerPixel = cgImage.bitsPerPixel / cgImage.bitsPerComponent
        let destBytesPerRow = width * bytesPerPixel

        guard let colorspace = CGColorSpace(name: CGColorSpace.linearSRGB)
        else {
            throw GPUError.generic
        }
        guard
              let cgContext = CGContext(
                data: nil,
                width: 480,
                height: 640,
                bitsPerComponent: cgImage.bitsPerComponent,
                bytesPerRow: destBytesPerRow,
                space: colorspace,
                bitmapInfo: cgImage.alphaInfo.rawValue
              )
        else {
            throw GPUError.generic
        }

        cgContext.interpolationQuality = .default
        cgContext.draw(cgImage, in: CGRect(x: 0, y: 0, width: 480, height: 640))

        guard let cgImage = cgContext.makeImage()
        else {
            throw GPUError.generic
        }

        return try loader.newTexture(cgImage: cgImage)
    }

    func texture(pngOf image: UIImage) throws -> MTLTexture {
        guard let data = image.pngData()
        else {
            throw GPUError.generic
        }
        return try loader.newTexture(data: data)
    }
}

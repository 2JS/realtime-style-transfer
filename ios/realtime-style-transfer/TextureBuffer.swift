import Metal
import CoreML


struct TextureBuffer {
    let height: Int
    let width: Int

    let buffer: MTLBuffer
    let texture: MTLTexture
    let data: Data

    var pixelFormat: MTLPixelFormat { texture.pixelFormat }

    var mlmultiarray: MLMultiArray? {
        get {
            try? MLMultiArray(
                dataPointer: buffer.contents(),
                shape: [1, height, width, pixelFormat.channels] as [NSNumber],
                dataType: .float32,
                strides: [height * width * pixelFormat.channels, width * pixelFormat.channels, pixelFormat.channels, 1] as [NSNumber],
                deallocator: .none
            )
        }
        set {
            guard let dataPointer = newValue?.dataPointer
            else { return }
//            Data(bytesNoCopy: dataPointer, count: height * width * pixelFormat.bytesPerPixel, deallocator: .none)
            memcpy(buffer.contents(), dataPointer, buffer.length)
        }
    }

    init?(
        device: MTLDevice,
        height: Int,
        width: Int,
        pixelFormat: MTLPixelFormat,
        usage: MTLTextureUsage
    ) {
        let size = width * height * pixelFormat.bytesPerPixel
        guard let buffer = device.makeBuffer(length: size.aligned(to: Int(getpagesize())), options: [.storageModeShared]),
              let texture = buffer.makeTexture(descriptor: MTLTextureDescriptor().then {
                  $0.height = height
                  $0.width = width
                  $0.pixelFormat = pixelFormat
                  $0.usage = usage
                  $0.resourceOptions = .storageModeShared
              }, offset: 0, bytesPerRow: width * pixelFormat.bytesPerPixel)
        else {
            return nil
        }

        self.height = height
        self.width = width
        self.buffer = buffer
        self.texture = texture
        self.data = Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none)
    }
}

import CoreVideo
import Metal

extension MTLTexture {
    var pixelBuffer: CVPixelBuffer? {
        var outPixelbuffer: CVPixelBuffer?
        guard let datas = self.buffer?.contents()
        else {
            return nil
        }
        CVPixelBufferCreateWithBytes(kCFAllocatorDefault, width,
        height, kCVPixelFormatType_32BGRA, datas,
        bufferBytesPerRow, nil, nil, nil, &outPixelbuffer);
        return outPixelbuffer
    }
}

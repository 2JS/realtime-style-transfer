import AVFoundation
import Combine

class VideoStream: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let queue = DispatchQueue.global(qos: .userInitiated)
    let publisher = PassthroughSubject<CMSampleBuffer, Never>()

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        publisher.send(sampleBuffer)
    }
}

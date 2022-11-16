import CoreML
import UIKit

class ViewController: UIViewController {

//    let encoder: adain_vgg = {
//        let config = MLModelConfiguration()
//        config.allowLowPrecisionAccumulationOnGPU = true
//        config.computeUnits = .all
//        return try! adain_vgg(configuration: config)
//    }()
//
//    let decoder: adain_dec = {
//        let config = MLModelConfiguration()
//        config.allowLowPrecisionAccumulationOnGPU = true
//        config.computeUnits = .all
//        return try! adain_dec(configuration: config)
//    }()

    private let cameraViewController = CameraViewController()

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.

        addChild(cameraViewController)

        view.addSubview(cameraViewController.view)
        cameraViewController.view.frame = view.bounds
    }

    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)

//        var data = Data(repeating: 0, count: 3*256*256*MemoryLayout<Float32>.stride)
//        let shape: [NSNumber] = [1, 3, 256, 256]
//        let stride = MemoryLayout<Float32>.stride
//        let strides: [NSNumber] = [3*256*256*stride, 256*256*stride, 256*stride, stride] as [NSNumber]
//        let array: MLMultiArray = data.withUnsafeMutableBytes { pointer -> MLMultiArray in
//            return try! MLMultiArray(dataPointer: pointer, shape: shape, dataType: .float32, strides: strides)
//        }
//        let input = adain_vggInput(input_1: array)
//
//        for _ in 0..<5 {
//            let latent = Log.bench("Encode") { (try! encoder.prediction(input: input)).var_202 }
////            let output = Log.bench("Decode") { (try! decoder.prediction(input_1: latent)).var_174 }
//        }
    }
}


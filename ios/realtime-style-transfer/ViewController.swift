import CoreML
import UIKit

class ViewController: UIViewController {
    private let cameraViewController = CameraViewController()

    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.

        addChild(cameraViewController)

        view.addSubview(cameraViewController.view)
        cameraViewController.view.frame = view.bounds
    }
}


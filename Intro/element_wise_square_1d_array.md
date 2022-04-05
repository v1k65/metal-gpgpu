### Given a 1D array of Floats or Int, generate element-wise square.
Example: Input -> [1, 5, -3], Output -> [1, 25, 9]


First lets add a helper class `GpuFloatArray1d` to easily read and write on both CPU and GPU. Similar to Swift `Array` but memory is accessible to both GPU and CPU.
```
import Metal

typealias GpuFloatArray1d = GpuArray1d<Float>

class GpuArray1d<T>  {

	let buffer: MTLBuffer
	let count: Int

	private let _content: UnsafeMutablePointer<T>

	init(count: Int, device: MTLDevice, label: String? = nil) {
		self.buffer = device.makeBuffer(length: MemoryLayout<T>.stride * count)!
		self.count = count

		self._content = self.buffer.contents().bindMemory(to: T.self, capacity: count)

		buffer.label = label
	}

	subscript(idx: Int) -> T {
		get { _content[idx] }
		set { _content[idx] = newValue }
	}

	func resetAll(with val: T) {
		for idx in 0..<count {
			_content[idx] = val
		}
	}
}

extension GpuFloatArray1d : CustomStringConvertible {

	var description: String {
		var str =  buffer.label == nil ? "[ " : buffer.label! + " [ "
		for idx in 0..<count {
			str.append(self[idx].description + ", ")
		}
		str.append("]")

		return str
	}
}
```

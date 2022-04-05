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

kernel is below, each thread will read and square the element.
```
#include <metal_stdlib>
using namespace metal;

kernel void array_square(device float *elments 		[[ buffer(0) ]],
												 device float *output			[[ buffer(1) ]],
												 uint thread_idx				  [[ thread_position_in_grid ]]
												 ) {
	output[thread_idx] = elments[thread_idx] * elments[thread_idx];
}
```

Finally the host code on CUP to run the kernel
```
import MetalKit
import Metal

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!

let library = device.makeDefaultLibrary()!
let pipeline = try! device.makeComputePipelineState(function: library.makeFunction(name: "array_square")!)

let inputArray: GpuFloatArray1d = GpuFloatArray1d(count: 100, device: device, label: "input")
for idx in 0..<inputArray.count {
	inputArray[idx] = Float(idx)
}

let outputArray: GpuFloatArray1d = GpuFloatArray1d(count: inputArray.count, device: device, label: "output")

let computBuffer = queue.makeCommandBuffer()!

let commandEncoder = computBuffer.makeComputeCommandEncoder()!
commandEncoder.setComputePipelineState(pipeline)
commandEncoder.setBuffer(inputArray.buffer, offset: 0, index: 0)
commandEncoder.setBuffer(outputArray.buffer, offset: 0, index: 1)

commandEncoder.dispatchThreads(MTLSizeMake(inputArray.count, 1, 1),
															 threadsPerThreadgroup: MTLSizeMake(pipeline.threadExecutionWidth, 1, 1))
commandEncoder.endEncoding()

computBuffer.commit()
computBuffer.waitUntilCompleted()

print(inputArray)
print(outputArray)
```

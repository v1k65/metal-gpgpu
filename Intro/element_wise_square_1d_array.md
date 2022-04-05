### Given a 1D array of Floats or Int, generate element-wise square.
Example: Input -> [1, 5, -3], Output -> [1, 25, 9]


First lets add a helper class `GpuFloatArray1d`, similar to Swift `Array` but memory is accessible to both GPU and CPU.
``` swift
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
``` C++
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
``` Swift
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

Output:
```
input [ 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 63.0, 64.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, ]

output [ 0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0, 100.0, 121.0, 144.0, 169.0, 196.0, 225.0, 256.0, 289.0, 324.0, 361.0, 400.0, 441.0, 484.0, 529.0, 576.0, 625.0, 676.0, 729.0, 784.0, 841.0, 900.0, 961.0, 1024.0, 1089.0, 1156.0, 1225.0, 1296.0, 1369.0, 1444.0, 1521.0, 1600.0, 1681.0, 1764.0, 1849.0, 1936.0, 2025.0, 2116.0, 2209.0, 2304.0, 2401.0, 2500.0, 2601.0, 2704.0, 2809.0, 2916.0, 3025.0, 3136.0, 3249.0, 3364.0, 3481.0, 3600.0, 3721.0, 3844.0, 3969.0, 4096.0, 4225.0, 4356.0, 4489.0, 4624.0, 4761.0, 4900.0, 5041.0, 5184.0, 5329.0, 5476.0, 5625.0, 5776.0, 5929.0, 6084.0, 6241.0, 6400.0, 6561.0, 6724.0, 6889.0, 7056.0, 7225.0, 7396.0, 7569.0, 7744.0, 7921.0, 8100.0, 8281.0, 8464.0, 8649.0, 8836.0, 9025.0, 9216.0, 9409.0, 9604.0, 9801.0, ]
```

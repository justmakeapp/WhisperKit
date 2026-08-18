//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2026 Argmax, Inc. All rights reserved.
//  Created by Codex on 18/08/2026.

import CoreML
import XCTest

@testable import ArgmaxCore

final class MLMultiArrayExtensionsTests: XCTestCase {
    func testInitializesDoubleValues() throws {
        let array = try MLMultiArray(shape: [2, 3], dataType: .double, initialValue: Double(2.25))

        XCTAssertEqual(array.shape, [2, 3])
        XCTAssertEqual(values(in: array, as: Double.self), Array(repeating: 2.25, count: 6))
    }

    func testInitializesFloat32Values() throws {
        let array = try MLMultiArray(shape: [2, 2, 3], dataType: .float32, initialValue: Float(-1.5))

        XCTAssertEqual(array.shape, [2, 2, 3])
        XCTAssertEqual(values(in: array, as: Float.self), Array(repeating: -1.5, count: 12))
    }

    func testInitializesFloat16Values() throws {
        let initialValue = FloatType(1.5)
        let array = try MLMultiArray(shape: [2, 3, 4], dataType: .float16, initialValue: initialValue)

        XCTAssertEqual(array.shape, [2, 3, 4])
        XCTAssertEqual(float16BitPatterns(in: array), Array(repeating: 0x3e00, count: 24))
    }

    func testInitializesDecoderFloat16Values() throws {
        let cases: [(value: FloatType, expectedBitPattern: UInt16)] = [
            (0, 0x0000),
            (-10_000, 0xf0e2),
        ]

        for testCase in cases {
            let array = try MLMultiArray(shape: [2, 3], dataType: .float16, initialValue: testCase.value)

            XCTAssertEqual(
                float16BitPatterns(in: array),
                Array(repeating: testCase.expectedBitPattern, count: 6)
            )
        }
    }

    func testInitializesInt32Values() throws {
        let array = try MLMultiArray(shape: [3, 2], dataType: .int32, initialValue: Int32(-7))

        XCTAssertEqual(array.shape, [3, 2])
        XCTAssertEqual(values(in: array, as: Int32.self), Array(repeating: -7, count: 6))
    }

    #if compiler(>=6.2)
    @available(macOS 26.0, iOS 26.0, watchOS 26.0, visionOS 26.0, tvOS 26.0, *)
    func testInitializesInt8Values() throws {
        let array = try MLMultiArray(shape: [2, 3], dataType: .int8, initialValue: Int8(5))

        XCTAssertEqual(array.shape, [2, 3])
        XCTAssertEqual(values(in: array, as: Int8.self), Array(repeating: 5, count: 6))
    }
    #endif

    private func values<Scalar: MLShapedArrayScalar>(
        in array: MLMultiArray,
        as type: Scalar.Type
    ) -> [Scalar] {
        array.withUnsafeBufferPointer(ofType: type) { buffer in
            Array(buffer.prefix(array.count))
        }
    }

    private func float16BitPatterns(in array: MLMultiArray) -> [UInt16] {
        array.withUnsafeBytes { buffer in
            let shape = array.shape.map(\.intValue)
            let strides = array.strides.map(\.intValue)

            return (0..<array.count).map { linearIndex in
                var remainingIndex = linearIndex
                var elementOffset = 0

                for dimension in shape.indices.reversed() {
                    let dimensionIndex = remainingIndex % shape[dimension]
                    remainingIndex /= shape[dimension]
                    elementOffset += dimensionIndex * strides[dimension]
                }

                let byteOffset = elementOffset * MemoryLayout<UInt16>.stride
                return UInt16(buffer[byteOffset]) | UInt16(buffer[byteOffset + 1]) << 8
            }
        }
    }
}

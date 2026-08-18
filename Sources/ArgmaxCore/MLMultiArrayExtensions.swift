//  For licensing see accompanying LICENSE.md file.
//  Copyright © 2024 Argmax, Inc. All rights reserved.

import CoreML

// MARK: - MLMultiArray Creation

public extension MLMultiArray {
    /// Creates an MLMultiArray pre-filled with an initial value.
    /// Uses IOSurface-backed storage for float16 arrays except on x86_64.
    convenience init(shape: [NSNumber], dataType: MLMultiArrayDataType, initialValue: Any) throws {
        switch dataType {
            case .float16:
                #if arch(x86_64)
                try self.init(shape: shape, dataType: dataType)
                #else
                guard let pixelBuffer = Self.pixelBuffer(for: shape) else {
                    throw MLMultiArrayCreationError.pixelBufferFailed
                }
                self.init(pixelBuffer: pixelBuffer, shape: shape)
                #endif
            default:
                try self.init(shape: shape, dataType: dataType)
        }

        switch dataType {
            case .double:
                if let value = initialValue as? Double {
                    fillBuffer(with: value)
                }
            case .float32:
                if let value = initialValue as? Float {
                    fillBuffer(with: value)
                }
            case .float16:
                if let value = initialValue as? FloatType {
                    fillFloat16Buffer(with: Self.float16BitPattern(for: Float(value)))
                }
            case .int32:
                if let value = initialValue as? Int32 {
                    fillBuffer(with: value)
                }
            #if compiler(>=6.2)
            case .int8:
                if #available(macOS 26.0, iOS 26.0, watchOS 26.0, visionOS 26.0, tvOS 26.0, *),
                   let value = initialValue as? Int8 {
                    fillBuffer(with: value)
                }
            #endif
            @unknown default:
                break
        }
    }

    private func fillBuffer<Scalar: MLShapedArrayScalar>(with value: Scalar) {
        withUnsafeMutableBufferPointer(ofType: Scalar.self) { buffer, _ in
            for index in buffer.indices {
                buffer[index] = value
            }
        }
    }

    private func fillFloat16Buffer(with bitPattern: UInt16) {
        var bitPattern = bitPattern
        withUnsafeMutableBytes { buffer, _ in
            Swift.withUnsafeBytes(of: &bitPattern) { valueBytes in
                for offset in stride(from: 0, to: buffer.count - 1, by: MemoryLayout<UInt16>.stride) {
                    buffer[offset] = valueBytes[0]
                    buffer[offset + 1] = valueBytes[1]
                }
            }
        }
    }

    private static func float16BitPattern(for value: Float) -> UInt16 {
        let bitPattern = value.bitPattern
        let sign = UInt16(truncatingIfNeeded: bitPattern >> 16) & 0x8000
        let exponent = Int((bitPattern >> 23) & 0xff)
        let significand = bitPattern & 0x7fffff

        if exponent == 0xff {
            guard significand != 0 else { return sign | 0x7c00 }
            let payload = UInt16(truncatingIfNeeded: significand >> 13)
            return sign | 0x7c00 | payload | 0x0200
        }

        let float16Exponent = exponent - 127 + 15
        if float16Exponent >= 0x1f {
            return sign | 0x7c00
        }

        if float16Exponent <= 0 {
            guard float16Exponent >= -10 else { return sign }
            let significandWithHiddenBit = significand | 0x800000
            let shift = UInt32(14 - float16Exponent)
            let rounded = roundToNearestEven(significandWithHiddenBit, shift: shift)
            return sign | UInt16(truncatingIfNeeded: rounded)
        }

        let roundedSignificand = roundToNearestEven(significand, shift: 13)
        let magnitude = (UInt32(float16Exponent) << 10) + roundedSignificand
        return sign | UInt16(truncatingIfNeeded: magnitude)
    }

    private static func roundToNearestEven(_ value: UInt32, shift: UInt32) -> UInt32 {
        let truncated = value >> shift
        let remainderMask = (UInt32(1) << shift) - 1
        let remainder = value & remainderMask
        let halfway = UInt32(1) << (shift - 1)
        let shouldRoundUp = remainder > halfway || (remainder == halfway && truncated & 1 == 1)
        return truncated + (shouldRoundUp ? 1 : 0)
    }

    /// Creates an MLMultiArray from an [Int] array.
    /// Values are stored in the last dimension (default is dims=1).
    static func from(_ array: [Int], dims: Int = 1) throws -> MLMultiArray {
        var shape = Array(repeating: 1, count: dims)
        shape[shape.count - 1] = array.count
        let output = try MLMultiArray(shape: shape as [NSNumber], dataType: .int32)
        let pointer = UnsafeMutablePointer<Int32>(OpaquePointer(output.dataPointer))
        for (i, item) in array.enumerated() {
            pointer[i] = Int32(item)
        }
        return output
    }
}

// MARK: - MLMultiArray Indexing & Fill

public extension MLMultiArray {
    /// Computes the linear offset from multi-dimensional indices using strides.
    @inline(__always)
    func linearOffset(for index: [Int], strides strideInts: [Int]? = nil) -> Int {
        var linearOffset = 0
        let strideInts = strideInts ?? strides.map { $0.intValue }
        for (dimension, stride) in zip(index, strideInts) {
            linearOffset += dimension * stride
        }
        return linearOffset
    }

    @available(*, deprecated, message: "Use linearOffset(for: [Int], strides:) instead.")
    @inline(__always)
    func linearOffset(for index: [NSNumber], strides strideInts: [Int]? = nil) -> Int {
        linearOffset(for: index.map(\.intValue), strides: strideInts)
    }

    /// Fills a range of indices in the last dimension with a value.
    /// Requires shape [1, 1, n].
    func fillLastDimension(indexes: Range<Int>, with value: FloatType) {
        precondition(shape.count == 3 && shape[0] == 1 && shape[1] == 1, "Must have [1, 1, n] shape")
        withUnsafeMutableBufferPointer(ofType: FloatType.self) { ptr, strides in
            for index in indexes {
                ptr[index * strides[2]] = value
            }
        }
    }

    /// Fills specific multi-dimensional indices with a value.
    func fill<Value>(indexes: [[Int]], with value: Value) {
        let pointer = UnsafeMutablePointer<Value>(OpaquePointer(dataPointer))
        let strideInts = strides.map { $0.intValue }
        let shapeInts = shape.map { $0.intValue }
        for index in indexes {
            guard index.count == shapeInts.count,
                  zip(index, shapeInts).allSatisfy({ $0 >= 0 && $0 < $1 }) else {
                continue
            }
            let linearOffset = linearOffset(for: index, strides: strideInts)
            pointer[linearOffset] = value
        }
    }

    @available(*, deprecated, message: "Use fill(indexes: [[Int]], with:) instead.")
    func fill<Value>(indexes: [[NSNumber]], with value: Value) {
        fill(indexes: indexes.map { $0.map(\.intValue) }, with: value)
    }
}

// MARK: - IOSurface-backed Pixel Buffer

extension MLMultiArray {
    /// Creates a CVPixelBuffer suitable for float16 IOSurface-backed MLMultiArrays.
    public class func pixelBuffer(for shape: [NSNumber]) -> CVPixelBuffer? {
        guard let width = shape.last?.intValue else { return nil }
        let height = shape[0..<shape.count - 1].reduce(1) { $0 * $1.intValue }

        var pixelBuffer: CVPixelBuffer?
        let createReturn = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_OneComponent16Half,
            [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary,
            &pixelBuffer
        )
        guard createReturn == kCVReturnSuccess else { return nil }
        return pixelBuffer
    }
}

// MARK: - Error

/// Error thrown when MLMultiArray creation fails.
@frozen
public enum MLMultiArrayCreationError: Error, LocalizedError {
    case pixelBufferFailed

    public var errorDescription: String? {
        switch self {
        case .pixelBufferFailed:
            return "Failed to create IOSurface-backed pixel buffer for MLMultiArray"
        }
    }
}

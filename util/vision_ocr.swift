import Foundation
import AppKit
import Vision

struct OCRItem: Codable {
    let text: String
    let confidence: Double
    let bbox: [Double]
}

struct OCRPayload: Codable {
    let width: Double
    let height: Double
    let items: [OCRItem]
}

func normalizedVisionLanguages(_ raw: String) -> [String] {
    let parts = raw.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
    let mapping: [String: String] = [
        "ch_sim": "zh-Hans",
        "ch_tra": "zh-Hant",
        "en": "en-US",
        "ja": "ja-JP",
        "ko": "ko-KR"
    ]
    let languages = parts.compactMap { mapping[$0] }
    return languages.isEmpty ? ["zh-Hans", "en-US"] : Array(NSOrderedSet(array: languages)) as! [String]
}

func loadCGImage(from path: String) throws -> CGImage {
    let url = URL(fileURLWithPath: path)
    guard let image = NSImage(contentsOf: url) else {
        throw NSError(domain: "vision_ocr", code: 1, userInfo: [NSLocalizedDescriptionKey: "Failed to load image"])
    }

    var proposedRect = NSRect(origin: .zero, size: image.size)
    guard let cgImage = image.cgImage(forProposedRect: &proposedRect, context: nil, hints: nil) else {
        throw NSError(domain: "vision_ocr", code: 2, userInfo: [NSLocalizedDescriptionKey: "Failed to decode CGImage"])
    }
    return cgImage
}

let args = CommandLine.arguments
if args.count < 2 {
    fputs("usage: vision_ocr <image-path> [langs]\n", stderr)
    exit(2)
}

let imagePath = args[1]
let langs = normalizedVisionLanguages(args.count >= 3 ? args[2] : "ch_sim,en")

let cgImage: CGImage

do {
    cgImage = try loadCGImage(from: imagePath)
} catch {
    fputs("\(error.localizedDescription)\n", stderr)
    exit(3)
}

let width = Double(cgImage.width)
let height = Double(cgImage.height)
var observations: [VNRecognizedTextObservation] = []

let request = VNRecognizeTextRequest { request, error in
    if let error = error {
        fputs("\(error.localizedDescription)\n", stderr)
        exit(4)
    }
    observations = (request.results as? [VNRecognizedTextObservation]) ?? []
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
request.recognitionLanguages = langs
request.minimumTextHeight = 0.004
if #available(macOS 13.0, *) {
    request.automaticallyDetectsLanguage = false
}

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fputs("\(error.localizedDescription)\n", stderr)
    exit(5)
}

let items = observations.compactMap { observation -> OCRItem? in
    guard let candidate = observation.topCandidates(1).first else { return nil }
    let text = candidate.string.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { return nil }

    let box = observation.boundingBox
    let x1 = box.minX * width
    let y1 = (1.0 - box.maxY) * height
    let x2 = box.maxX * width
    let y2 = (1.0 - box.minY) * height
    return OCRItem(text: text, confidence: Double(candidate.confidence), bbox: [x1, y1, x2, y2])
}.sorted {
    let ay = $0.bbox[1]
    let by = $1.bbox[1]
    if abs(ay - by) > 6 {
        return ay < by
    }
    return $0.bbox[0] < $1.bbox[0]
}

let payload = OCRPayload(width: width, height: height, items: items)
let encoder = JSONEncoder()
encoder.outputFormatting = []
let data = try encoder.encode(payload)
FileHandle.standardOutput.write(data)

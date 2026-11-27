// --- Day 2: I Was Told There Would Be No Math ---

import Foundation

let inputFile = URL(filePath: CommandLine.arguments[1])
let file = try FileHandle(forReadingFrom: inputFile)
defer { try? file.close() }  // todo: find out what this does

for try await line in file.bytes.lines {
    print(line)
}
// swift-tools-version: 6.0
import PackageDescription
import Foundation

let demoInfo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("native/DeskPetDemo-Info.plist").path

let package = Package(
    name: "DeskPetPipeline",
    platforms: [.macOS(.v13)],
    products: [.library(name: "DeskPetPipeline", targets: ["DeskPetPipeline"]),
               .executable(name: "deskpet-demo", targets: ["DeskPetDemo"])],
    dependencies: [.package(name: "DeskPetHardware", path: "../deskpet-hardware/swift")],
    targets: [.target(name: "CNativeInference", publicHeadersPath: "include"),
              .target(name: "DeskPetPipeline", dependencies: ["CNativeInference", .product(name: "DeskPetHardware", package: "DeskPetHardware")]),
              .executableTarget(name: "DeskPetDemo", dependencies: ["DeskPetPipeline"], linkerSettings: [
                .unsafeFlags(["-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker", demoInfo], .when(platforms: [.macOS]))
              ]),
              .testTarget(name: "DeskPetPipelineTests", dependencies: ["DeskPetPipeline"])]
)

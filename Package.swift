// swift-tools-version: 6.0
import PackageDescription
import Foundation

let demoInfo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("native/DeskPetDemo-Info.plist").path

let package = Package(
    name: "DeskPetPipeline",
    platforms: [.macOS(.v13)],
    products: [.library(name: "DeskPetPipeline", targets: ["DeskPetPipeline"]),
               .executable(name: "deskpet-demo", targets: ["DeskPetDemo"]),
               .library(name: "DeskPetHardwareIntegration", targets: ["DeskPetHardwareIntegration"]),
               .executable(name: "deskpet-board", targets: ["DeskPetBoard"])],
    dependencies: [.package(path: "../deskpet-hardware/swift")],
    targets: [.target(name: "DeskPetPipeline"),
              .executableTarget(name: "DeskPetDemo", dependencies: ["DeskPetPipeline"]),
              .target(name: "DeskPetHardwareIntegration", dependencies: ["DeskPetPipeline", .product(name: "DeskPetHardware", package: "swift")]),
              .executableTarget(name: "DeskPetBoard", dependencies: ["DeskPetHardwareIntegration"]),
              .testTarget(name: "DeskPetHardwareIntegrationTests", dependencies: ["DeskPetHardwareIntegration"]),
              .testTarget(name: "DeskPetPipelineTests", dependencies: ["DeskPetPipeline"])]
)

// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DeskPetPipeline",
    platforms: [.macOS(.v13)],
    products: [.library(name: "DeskPetPipeline", targets: ["DeskPetPipeline"]),
               .executable(name: "deskpet-demo", targets: ["DeskPetDemo"])],
    dependencies: [.package(name: "DeskPetHardware", path: "../deskpet-hardware/swift")],
    targets: [.target(name: "DeskPetPipeline", dependencies: [.product(name: "DeskPetHardware", package: "DeskPetHardware")]),
              .executableTarget(name: "DeskPetDemo", dependencies: ["DeskPetPipeline"]),
              .testTarget(name: "DeskPetPipelineTests", dependencies: ["DeskPetPipeline"])]
)

// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "DeskPetPipeline",
    platforms: [.macOS(.v13)],
    products: [.library(name: "DeskPetPipeline", targets: ["DeskPetPipeline"]),
               .executable(name: "deskpet-demo", targets: ["DeskPetDemo"])],
    targets: [.target(name: "DeskPetPipeline"),
              .executableTarget(name: "DeskPetDemo", dependencies: ["DeskPetPipeline"]),
              .testTarget(name: "DeskPetPipelineTests", dependencies: ["DeskPetPipeline"])]
)

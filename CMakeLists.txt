# Modified from Slicer's Default extension template; see LICENSES/Slicer.txt.
cmake_minimum_required(VERSION 3.28.0...3.28.0 FATAL_ERROR)

project(MONAIPhysio)

set(EXTENSION_HOMEPAGE "https://github.com/Project-MONAI/monai-physio-slicer")
set(EXTENSION_CATEGORY "Simulation")
set(EXTENSION_CONTRIBUTORS "Stephen Aylward")
set(EXTENSION_DESCRIPTION "Simulate physiological motion and connect 3D Slicer to NVIDIA Omniverse using MONAI-Physio: export surface or volumetric models as OpenUSD assets, and predict a respiratory cycle of lung motion (surfaces, 4D-CT and animated USD) from a chest CT. Setup downloads PyTorch and MONAI-Physio; respiratory simulation needs a CUDA GPU.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/Project-MONAI/monai-physio-slicer/main/MONAIPhysioExportUSD/Resources/Icons/MONAIPhysioExportUSD.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/Project-MONAI/monai-physio-slicer/main/docs/images/export-usd.png https://raw.githubusercontent.com/Project-MONAI/monai-physio-slicer/main/docs/images/respiratory-simulation.png https://raw.githubusercontent.com/Project-MONAI/monai-physio-slicer/main/docs/images/workflow.png")
set(EXTENSION_DEPENDS "NA")
set(EXTENSION_STATUS "Experimental")

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

add_subdirectory(MONAIPhysioExportUSD)
add_subdirectory(MONAIPhysioRespiratorySimulation)

include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})

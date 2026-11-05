# Modified from Slicer's Scripted module template; see LICENSES/Slicer.txt.
set(MODULE_NAME MONAIPhysioExportUSD)

# MONAIPhysioLib holds the helper class and worker used by this module.
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  MONAIPhysioLib/__init__.py
  MONAIPhysioLib/helper.py
  MONAIPhysioLib/snapshot.py
  MONAIPhysioLib/export_worker.py
  MONAIPhysioLib/respiratory_simulation_worker.py
  MONAIPhysioLib/runtime.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/requirements.txt
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()

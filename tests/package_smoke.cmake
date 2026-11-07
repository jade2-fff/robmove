function(run_checked)
    execute_process(COMMAND ${ARGV} RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "Command failed: ${ARGV}\n${output}\n${error}")
    endif()
endfunction()

set(prefix "${ROBMOVE_BUILD}/install-smoke")
set(consumer "${ROBMOVE_BUILD}/consumer-smoke")
run_checked("${CMAKE_COMMAND}" --install "${ROBMOVE_BUILD}" --prefix "${prefix}")
run_checked("${CMAKE_COMMAND}" -S "${ROBMOVE_SOURCE}/tests/consumer" -B "${consumer}"
    "-DCMAKE_PREFIX_PATH=${prefix}" "-DCMAKE_CXX_COMPILER=${ROBMOVE_COMPILER}")
run_checked("${CMAKE_COMMAND}" --build "${consumer}" --parallel 2)
run_checked("${consumer}/consumer")

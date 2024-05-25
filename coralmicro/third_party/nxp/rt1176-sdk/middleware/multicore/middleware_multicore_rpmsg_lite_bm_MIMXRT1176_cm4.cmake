include_guard(GLOBAL)
message("middleware_multicore_rpmsg_lite_bm component is included.")

if(CONFIG_USE_middleware_baremetal_MIMXRT1176_cm4)
target_sources(${MCUX_SDK_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/rpmsg_lite/lib/rpmsg_lite/porting/environment/rpmsg_env_bm.c
)
else()
    message(WARNING "please config middleware.baremetal_MIMXRT1176_cm4 first.")
endif()


target_include_directories(${MCUX_SDK_PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/rpmsg_lite/lib/include/environment/bm
)



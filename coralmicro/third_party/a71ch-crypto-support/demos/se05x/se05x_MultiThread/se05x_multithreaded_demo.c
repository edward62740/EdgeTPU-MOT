/* Copyright 2020 NXP
 *
 * SPDX-License-Identifier: Apache-2.0
 * */

/* ************************************************************************** */
/* Includes                                                                   */
/* ************************************************************************** */

#include <nxLog_App.h>

#include <sm_timer.h>
#include "se05x_multithreaded_demo.h"
#include "PlugAndTrust_Pkg_Ver.h"

#ifdef HAVE_KSDK
#include "ex_sss_main_inc_ksdk.h"
#endif

#if AX_EMBEDDED && USE_RTOS
#include "FreeRTOS.h"
#endif
/*******************************************************************************
 * Definitions
 ******************************************************************************/

#if defined(__linux__) && defined(T1oI2C)
#if SSS_HAVE_APPLET_SE05X_IOT
#include "ex_sss_main_inc_linux.h"
#endif
#endif

/* ************************************************************************** */
/* Global Variables                                                           */
/* ************************************************************************** */
#if (__GNUC__ && !AX_EMBEDDED)
pthread_t tid[2];
#endif
extern demoCtx_t guser_session_1;
extern demoCtx_t guser_session_2;
extern demoCtx_t gbaseSession;
const char *portName;
/*******************************************************************
* MACROS
*******************************************************************/
#if AX_EMBEDDED && USE_RTOS
#define FREERTOS_MULTITHREAD_TASK_PRIORITY (1)
#define FREERTOS_MULTITHREAD_TASK_STACK_SIZE 3000
#endif

//#include <ex_sss_main_inc.h>

/*******************************************************************************
 * AFR
 ******************************************************************************/
extern sss_status_t setup_se05x(void);
extern void cleanup_se05x(void);
extern void* Sign_Verify_eccsecp256r1_sha224_guser1(void *ptr);
extern void* Sign_Verify_eccsecp256r1_sha512_user2(void *ptr);

/*******************************************************************************
 * Code
 ******************************************************************************/
int main(int argc, const char *argv[])
{
#if AX_EMBEDDED
	 ex_sss_main_ksdk_bm();
#endif

#if defined(__linux__) && defined(T1oI2C) && SSS_HAVE_APPLET_SE05X_IOT
    ex_sss_main_linux_conf();
#endif // defined(__linux__) && defined(T1oI2C) && SSS_HAVE_APPLET_SE05X_IOT

    LOG_I(PLUGANDTRUST_PROD_NAME_VER_FULL);

#if AX_EMBEDDED
    portName = NULL;
#else
    sss_status_t status;
    status = ex_sss_boot_connectstring(argc, argv, &portName);
    if (kStatus_SSS_Success != status) {
        LOG_E("ex_sss_boot_connectstring Failed");
        return 1;
    }
#endif // AX_EMBEDDED

#if (__GNUC__ && !AX_EMBEDDED)
    int ret;
    sss_status_t st;
    st = setup_se05x();
    if (st != kStatus_SSS_Success) {
    	return 1;
    }
    ret = pthread_create(&(tid[0]), NULL, Sign_Verify_eccsecp256r1_sha224_guser1, (void *)&guser_session_1);
    if (ret != 0) {
        LOG_E("Thread can't be created");
        return 1;
    }
    ret = pthread_create(&(tid[1]), NULL, Sign_Verify_eccsecp256r1_sha512_user2, (void *)&guser_session_2);
    if (ret != 0) {
        LOG_E("Thread can't be created");
        return 1;
    }

    pthread_join(tid[1], NULL);
    pthread_join(tid[0], NULL);
    cleanup_se05x();

#elif AX_EMBEDDED && USE_RTOS
    if (xTaskCreate((TaskFunction_t)setup_se05x, "multithreadTestsetup", 3000, NULL, 2, NULL) != pdPASS) {
        LOG_E("Task creation failed!.");
        while(1);
    }
    if (xTaskCreate((TaskFunction_t)Sign_Verify_eccsecp256r1_sha224_guser1, "multithreadTest1", FREERTOS_MULTITHREAD_TASK_STACK_SIZE, (void *)&guser_session_1, FREERTOS_MULTITHREAD_TASK_PRIORITY, NULL) != pdPASS) {
        LOG_E("Task creation failed!.");
        while(1);
    }
    if (xTaskCreate((TaskFunction_t)Sign_Verify_eccsecp256r1_sha512_user2, "multithreadTest2", FREERTOS_MULTITHREAD_TASK_STACK_SIZE, (void *)&guser_session_2, FREERTOS_MULTITHREAD_TASK_PRIORITY, NULL) != pdPASS) {
        LOG_E("Task creation failed!.");
        while(1);
    }
    if (xTaskCreate((TaskFunction_t)cleanup_se05x, "multithreadTestcleanup", 2000, NULL, 1, NULL) != pdPASS) {
        LOG_E("Task creation failed!.");
        while(1);
    }

    vTaskStartScheduler();
    while(1); // Should never return here

#endif
    return 0;
}

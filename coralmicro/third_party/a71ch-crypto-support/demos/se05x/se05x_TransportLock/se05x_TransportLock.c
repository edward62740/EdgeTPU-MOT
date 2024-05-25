/* Copyright 2020 NXP
*
* SPDX-License-Identifier: Apache-2.0
*/

#include <nxLog_App.h>
#include <se05x_APDU.h>

#include "se05x_TransportAuth.h"

static ex_sss_boot_ctx_t gex_sss_transport_lock_boot_ctx;

#define EX_SSS_BOOT_PCONTEXT (&gex_sss_transport_lock_boot_ctx)
#define EX_SSS_BOOT_DO_ERASE 0
#define EX_SSS_BOOT_EXPOSE_ARGC_ARGV 0

static sss_status_t se05x_TransportLock(Se05xSession_t *pSe05xSession);

#include <ex_sss_main_inc.h>
sss_status_t ex_sss_entry(ex_sss_boot_ctx_t *pCtx)
{
    sss_status_t status           = kStatus_SSS_Fail;
    sss_se05x_session_t *pSession = (sss_se05x_session_t *)&pCtx->session;
    smStatus_t sw_status;
    Se05xSession_t *pSe05xSession;
    sss_session_t txSession = {0};
    sss_tunnel_t txTunnel   = {0};

    /*Create transport object in host*/
    status = se05x_TA_CreateHostKey(pCtx);
    if (kStatus_SSS_Success != status) {
        LOG_E("Failed to set transport object in host!!!");
        goto cleanup;
    }
    pSe05xSession = &pSession->s_ctx;
    /*Create transport object in SE*/
    sw_status = se05x_TA_CreateSETransportLockKey(pSe05xSession);
    if (sw_status == SM_ERR_CONDITIONS_OF_USE_NOT_SATISFIED) {
        LOG_W("Transport to se05x applet is Locked!!!");
    }

    status = se05x_TA_OpenAuthSession(pCtx, &txSession, &txTunnel);

    if (kStatus_SSS_Success == status) {
        pSe05xSession = &((sss_se05x_session_t *)&txSession)->s_ctx;
        status        = se05x_TransportLock(pSe05xSession);
    }

    sss_session_close(&txSession);

cleanup:
    if (kStatus_SSS_Success == status) {
        LOG_I("se05x_TransportLock Example Success !!!...");
    }
    else {
        LOG_E("se05x_TransportLock Example Failed !!!...");
    }
    return status;
}

static sss_status_t se05x_TransportLock(Se05xSession_t *pSe05xSession)
{
    sss_status_t status = kStatus_SSS_Fail;
    smStatus_t sw_status;
    /* Persistent Lock */
    SE05x_LockIndicator_t lockIndicator = kSE05x_LockIndicator_PERSISTENT_LOCK;
    /* Lock during Transport */
    SE05x_LockState_t lockState = kSE05x_LockState_LOCKED;

    sw_status = Se05x_API_SetLockState(pSe05xSession, lockIndicator, lockState);
    if (SM_OK != sw_status) {
        LOG_E("Failed Se05x_API_SetLockState");
        goto cleanup;
    }
    else {
        LOG_W("Se05x_API_SetLockState Transport Lock feature is success");
        status = kStatus_SSS_Success;
    }

cleanup:
    return status;
}

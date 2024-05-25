/* Copyright 2018,2019 NXP
* SPDX-License-Identifier: Apache-2.0
*/

#include <ex_sss.h>
#include <stdio.h>
#include <stdlib.h>

#include "sm_printf.h"
#include "sss_mbedtls.h"
#include "usecase.h"

/* clang-format off */
const uint8_t client_key[] = { \
0x30, 0x81, 0x87, 0x02, 0x01, 0x00, 0x30, 0x13, 0x06, 0x07, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02,
0x01, 0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07, 0x04, 0x6D, 0x30, 0x6B, 0x02,
0x01, 0x01, 0x04, 0x20, 0x84, 0x28, 0x64, 0x80, 0x9E, 0x35, 0x62, 0xD6, 0x5A, 0x19, 0x76, 0xE6,
0x2A, 0x10, 0x0B, 0x79, 0x2A, 0x9D, 0x1B, 0x5D, 0x73, 0xA4, 0x7D, 0x70, 0x77, 0x84, 0xE6, 0x58,
0x59, 0x54, 0x86, 0x12, 0xA1, 0x44, 0x03, 0x42, 0x00, 0x04, 0x25, 0xDA, 0x3D, 0x60, 0xA5, 0x09,
0xA6, 0x1E, 0x3D, 0xD4, 0x54, 0x8B, 0x88, 0x28, 0x67, 0xD7, 0xA2, 0x70, 0x06, 0x01, 0x29, 0xE7,
0xBA, 0x72, 0x2B, 0x6D, 0xDB, 0xE5, 0x59, 0x68, 0x3C, 0x7E, 0xBF, 0xF9, 0x03, 0xC5, 0x11, 0x89,
0xD4, 0x77, 0xE6, 0x1D, 0xC4, 0xA2, 0x8C, 0x66, 0x9F, 0x82, 0xCA, 0x6E, 0xDC, 0xAF, 0xC6, 0x54,
0xCA, 0xC4, 0xEF, 0xF2, 0x52, 0x30, 0xC3, 0x7C, 0x3C, 0xCF };
const uint8_t rootca_key[] = { \
0x04, 0x20, 0x41, 0x90, 0xFA, 0xA5, 0x77, 0xB7, 0xA8, 0x02, 0xB7, 0x19, 0x2D, 0xA2, 0xD9, 0x45,
0x34, 0x54, 0xC7, 0x10, 0x4B, 0x0D, 0x3A, 0xF8, 0xBB, 0x3E, 0x1B, 0x73, 0x21, 0x9C, 0xE2, 0xFD,
0xC3, 0x27, 0xE9, 0x96, 0x0F, 0xBE, 0xD6, 0x06, 0xDE, 0x0E, 0x05, 0xE1, 0x9F, 0x31, 0x64, 0x90,
0x8E, 0x20, 0x93, 0x4E, 0xD4, 0xB0, 0xD4, 0xCE, 0xC9, 0xA6, 0xFC, 0xBF, 0xFC, 0xFF, 0x68, 0x37,
0xA8 };

const uint8_t client_cer[] = { \
0x30, 0x82, 0x01, 0xbf, 0x30, 0x82, 0x01, 0x65, 0x02, 0x09, 0x00, 0x88, 0xb8, 0xb8, 0xbe, 0x10,
0x46, 0x0d, 0x5a, 0x30, 0x0a, 0x06, 0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02, 0x30,
0x81, 0x8c, 0x31, 0x0b, 0x30, 0x09, 0x06, 0x03, 0x55, 0x04, 0x06, 0x13, 0x02, 0x42, 0x45, 0x31,
0x16, 0x30, 0x14, 0x06, 0x03, 0x55, 0x04, 0x08, 0x0c, 0x0d, 0x56, 0x6c, 0x61, 0x61, 0x6d, 0x73,
0x42, 0x72, 0x61, 0x62, 0x61, 0x6e, 0x74, 0x31, 0x0f, 0x30, 0x0d, 0x06, 0x03, 0x55, 0x04, 0x07,
0x0c, 0x06, 0x4c, 0x65, 0x75, 0x76, 0x65, 0x6e, 0x31, 0x14, 0x30, 0x12, 0x06, 0x03, 0x55, 0x04,
0x0a, 0x0c, 0x0b, 0x4e, 0x58, 0x50, 0x2d, 0x44, 0x65, 0x6d, 0x6f, 0x2d, 0x43, 0x41, 0x31, 0x12,
0x30, 0x10, 0x06, 0x03, 0x55, 0x04, 0x0b, 0x0c, 0x09, 0x44, 0x65, 0x6d, 0x6f, 0x2d, 0x55, 0x6e,
0x69, 0x74, 0x31, 0x0f, 0x30, 0x0d, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c, 0x06, 0x64, 0x65, 0x6d,
0x6f, 0x43, 0x41, 0x31, 0x19, 0x30, 0x17, 0x06, 0x09, 0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01,
0x09, 0x01, 0x16, 0x0a, 0x64, 0x65, 0x6d, 0x6f, 0x43, 0x41, 0x40, 0x6e, 0x78, 0x70, 0x30, 0x1e,
0x17, 0x0d, 0x31, 0x37, 0x31, 0x32, 0x32, 0x31, 0x30, 0x39, 0x30, 0x36, 0x34, 0x36, 0x5a, 0x17,
0x0d, 0x32, 0x35, 0x30, 0x38, 0x32, 0x31, 0x30, 0x39, 0x30, 0x36, 0x34, 0x36, 0x5a, 0x30, 0x42,
0x31, 0x0b, 0x30, 0x09, 0x06, 0x03, 0x55, 0x04, 0x06, 0x13, 0x02, 0x42, 0x45, 0x31, 0x10, 0x30,
0x0e, 0x06, 0x03, 0x55, 0x04, 0x0a, 0x0c, 0x07, 0x4e, 0x58, 0x50, 0x44, 0x65, 0x6d, 0x6f, 0x31,
0x0d, 0x30, 0x0b, 0x06, 0x03, 0x55, 0x04, 0x0b, 0x0c, 0x04, 0x55, 0x6e, 0x69, 0x74, 0x31, 0x12,
0x30, 0x10, 0x06, 0x03, 0x55, 0x04, 0x03, 0x0c, 0x09, 0x6c, 0x6f, 0x63, 0x61, 0x6c, 0x68, 0x6f,
0x73, 0x74, 0x30, 0x59, 0x30, 0x13, 0x06, 0x07, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01, 0x06,
0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07, 0x03, 0x42, 0x00, 0x04, 0x25, 0xda, 0x3d,
0x60, 0xa5, 0x09, 0xa6, 0x1e, 0x3d, 0xd4, 0x54, 0x8b, 0x88, 0x28, 0x67, 0xd7, 0xa2, 0x70, 0x06,
0x01, 0x29, 0xe7, 0xba, 0x72, 0x2b, 0x6d, 0xdb, 0xe5, 0x59, 0x68, 0x3c, 0x7e, 0xbf, 0xf9, 0x03,
0xc5, 0x11, 0x89, 0xd4, 0x77, 0xe6, 0x1d, 0xc4, 0xa2, 0x8c, 0x66, 0x9f, 0x82, 0xca, 0x6e, 0xdc,
0xaf, 0xc6, 0x54, 0xca, 0xc4, 0xef, 0xf2, 0x52, 0x30, 0xc3, 0x7c, 0x3c, 0xcf, 0x30, 0x0a, 0x06,
0x08, 0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02, 0x03, 0x48, 0x00, 0x30, 0x45, 0x02, 0x20,
0x12, 0xc7, 0x3b, 0x30, 0x07, 0x6f, 0x82, 0xc2, 0xc5, 0xcf, 0x0d, 0xae, 0x8d, 0x4e, 0x5f, 0xd3,
0xb9, 0x50, 0xd1, 0xad, 0x69, 0xb6, 0xf1, 0xad, 0x04, 0x0b, 0xbe, 0x49, 0x08, 0xab, 0xea, 0x1d,
0x02, 0x21, 0x00, 0x97, 0xcc, 0x49, 0x98, 0xc0, 0x45, 0x75, 0x8c, 0x89, 0xa1, 0xff, 0x91, 0x26,
0x2f, 0xcf, 0xb1, 0xe6, 0x0a, 0x6d, 0x91, 0xf5, 0xe6, 0x1a, 0x09, 0x4a, 0x35, 0x8a, 0x46, 0x4c,
0x32, 0xbb, 0xc5 };

/* clang-format on */

#define SSS_PUBKEY_INDEX_CA 0x7DCCBB22              //(1u)
#define SSS_KEYPAIR_INDEX_CLIENT_PRIVATE 0x20181001 //(2u)
#define SSS_CERTIFICATE_INDEX 0x20181002            //(3u)

/*The size of the client certificate should be checked when script is used to store it in GP storage and updated here */
#define SIZE_CLIENT_CERTIFICATE 500

int main(int argc, char *argv[])
{
    int ret                       = 0;
    const bool useKeysFromSM      = true;
    int client_certificate_loaded = 0;
    sss_status_t status;
    uint8_t aclient_cer[SIZE_CLIENT_CERTIFICATE] = {0};
    sss_status_t connectStatus;

    connectStatus = SessionOpenPort(argv[1], 1); /*1 ==> debug reset*/
    if (connectStatus != kStatus_SSS_Success) {
        sm_printf(CONSOLE, "Connection failed. SW = 0x%04X\n", connectStatus);
    }

    //Provision the SE..

    status = sss_key_object_init(&gSSSExCtx.keyPair, &pCtx->ks);
    if (status != kStatus_SSS_Success) {
        printf(" sss_key_object_init for keyPair Failed...\n");
        return;
    }

    status = sss_key_object_allocate_handle(&gSSSExCtx.keyPair,
        SSS_KEYPAIR_INDEX_CLIENT_PRIVATE,
        kSSS_KeyPart_Pair,
        kSSS_CipherType_EC_NIST_P,
        sizeof(client_key),
        kKeyObject_Mode_Persistent);
    if (status != kStatus_SSS_Success) {
        printf(" sss_key_object_allocate_handle for keyPair Failed...\n");
        return;
    }

    status = sss_key_store_set_key(&pCtx->ks, &gSSSExCtx.keyPair, client_key, sizeof(client_key), 256, NULL, 0);
    if (status != kStatus_SSS_Success) {
        printf(" sss_key_store_set_key  for keyPair Failed...\n");
        return;
    }

    status = sss_key_object_init(&gSSSExCtx.extPubkey, &pCtx->ks);
    if (status != kStatus_SSS_Success) {
        printf(" sss_key_object_init for Pub key Failed...\n");
        return;
    }

    status = sss_key_object_allocate_handle(&gSSSExCtx.extPubkey,
        SSS_PUBKEY_INDEX_CA,
        kSSS_KeyPart_Public,
        kSSS_CipherType_EC_NIST_P,
        sizeof(rootca_key),
        kKeyObject_Mode_Persistent);
    if (status != kStatus_SSS_Success) {
        printf(" sss_key_object_allocate_handle for Pub key Failed...\n");
        return;
    }

    status = sss_key_store_set_key(&pCtx->ks, &gSSSExCtx.extPubkey, rootca_key, sizeof(rootca_key), 256, NULL, 0);
    if (status != kStatus_SSS_Success) {
        printf(" sss_key_store_set_key for Pub key Failed...\n");
        return;
    }

    status = sss_key_object_init(&gSSSExCtx.clientCert, &pCtx->ks);
    if (status != kStatus_SSS_Success) {
        printf(" sss_key_object_init for Pub key Failed...\n");
        return;
    }

    status = sss_key_object_allocate_handle(&gSSSExCtx.clientCert,
        SSS_CERTIFICATE_INDEX,
        kSSS_KeyPart_Default,
        kSSS_CipherType_Binary,
        sizeof(client_cer),
        kKeyObject_Mode_Persistent);
    if (status != kStatus_SSS_Success) {
        printf(" sss_key_object_allocate_handle Failed!!!");
        return;
    }

    status = sss_key_store_set_key(
        &pCtx->ks, &gSSSExCtx.clientCert, client_cer, sizeof(client_cer), sizeof(client_cer) * 8, NULL, 0);
    if (status != kStatus_SSS_Success) {
        printf(" Store Certificate Failed!!!");
        return;
    }

    ex_SessionClose();
    return;
}

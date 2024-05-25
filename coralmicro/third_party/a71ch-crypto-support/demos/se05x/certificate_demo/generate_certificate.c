/* Copyright 2019 NXP
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ex_sss_boot.h>
#include <fsl_sss_se05x_apis.h>
#include <nxLog_App.h>
#include <openssl/asn1t.h>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <se05x_APDU.h>
#include <se05x_const.h>
#include <se05x_ecc_curves.h>
#include <se05x_ecc_curves_values.h>
#include <se05x_tlv.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "certificate.h"

static ex_sss_boot_ctx_t gex_sss_gen_cert;

#define EX_SSS_BOOT_PCONTEXT (&gex_sss_gen_cert)
#define EX_SSS_BOOT_DO_ERASE 0
#define EX_SSS_BOOT_EXPOSE_ARGC_ARGV 0

#include <ex_sss_main_inc.h>

sss_status_t ex_sss_entry(ex_sss_boot_ctx_t *pCtx)
{
    sss_status_t status = kStatus_SSS_Fail;
    sss_object_t obj;

    X509 *certificate = X509_new();

    status = sss_key_object_init(&obj, &pCtx->ks);
    if (status != kStatus_SSS_Success) {
        LOG_E("sss_key_object_init Failed!!!");
        return status;
    }

    if (!X509_set_version(certificate, 2)) {
        return status;
    }

    ASN1_INTEGER *serialNumber = ASN1_INTEGER_new();
    if (!ASN1_INTEGER_set(serialNumber, 1) || !X509_set_serialNumber(certificate, serialNumber)) {
        return status;
    }

    X509_NAME *subjectName = X509_NAME_new();
    if (!X509_NAME_add_entry_by_txt(subjectName,
            "CN",
            MBSTRING_ASC,
            (const unsigned char *)"selfSignedCA",
            -1 /* len */,
            -1 /* loc */,
            0 /* set */) ||
        !X509_set_subject_name(certificate, subjectName)) {
        return status;
    }

    if (!X509_set_issuer_name(certificate, subjectName)) {
        return status;
    }

    ASN1_TIME *notBefore  = ASN1_TIME_new();
    time_t activeDateTime = time(NULL);
    if (!ASN1_TIME_set(notBefore, activeDateTime) || !X509_set_notBefore(certificate, notBefore)) {
        return status;
    }

    ASN1_TIME *notAfter = ASN1_TIME_new();
    time_t notAfterTime = time(NULL);
    if (!ASN1_TIME_set(notAfter, notAfterTime * 2) || !X509_set_notAfter(certificate, notAfter)) {
        return status;
    }

    /*Generate a keypair*/
    uint8_t key[550];
    size_t keybytelen = sizeof(key);
    size_t keybitlen  = keybytelen * 8;
    uint32_t keyId    = CERTIFICATE_KEY_ID;
    status            = sss_key_object_get_handle(&obj, keyId);
    status            = sss_key_store_get_key(&pCtx->ks, &obj, key, &keybytelen, &keybitlen);
    LOG_I("sss_key_store_get_key status %x", status);
    if (status != kStatus_SSS_Success)
        return status;

    status   = kStatus_SSS_Fail;
    BIO *bio = BIO_new_mem_buf(key, (int)sizeof(key));
    if (bio == NULL) {
        return status;
    }

    EVP_PKEY *pKey = d2i_PUBKEY_bio(bio, NULL);
    if (pKey == NULL) {
        return status;
    }
    /*Add public key*/

    if (!X509_set_pubkey(certificate, pKey)) {
        return status;
    }

    X509V3_CTX x509v3_ctx = {0};

    char *subjectAlternateName    = "email:d:NXP-A71CH-D:377813426378607196897279";
    char *basicConstraints        = "CA:TRUE";
    X509_EXTENSION *extension_SAN = X509V3_EXT_nconf_nid(NULL, &x509v3_ctx, NID_subject_alt_name, subjectAlternateName);
    if (!X509_add_ext(certificate, extension_SAN, -1)) {
        return status;
    }
    X509_EXTENSION *extension_BasicConstraints =
        X509V3_EXT_nconf_nid(NULL, &x509v3_ctx, NID_basic_constraints, basicConstraints);
    if (!X509_add_ext(certificate, extension_BasicConstraints, -1)) {
        return status;
    }

    int type = NID_ecdsa_with_SHA256;
#if (OPENSSL_VERSION_NUMBER < 0x10100000L)
    X509_ALGOR *tbs_algo = certificate->cert_info->signature;
    X509_ALGOR *algo     = certificate->sig_alg;
    X509_ALGOR_set0(algo, OBJ_nid2obj(type), V_ASN1_NULL, NULL);
#else
    const X509_ALGOR *tmp_tbs_algo = X509_get0_tbs_sigalg((const X509 *)certificate);
    X509_ALGOR *tbs_algo           = (X509_ALGOR *)tmp_tbs_algo;
#endif
    X509_ALGOR_set0(tbs_algo, OBJ_nid2obj(type), V_ASN1_NULL, NULL);

    /*Convert to DER format to sign*/
    int len = 0;

    uint8_t tbs_bytes[500];
    uint8_t *p_tbs_bytes = &tbs_bytes[0];

#if (OPENSSL_VERSION_NUMBER < 0x10100000L)
    X509_CINF *cinf = certificate->cert_info;
    len             = i2d_X509_CINF(cinf, &p_tbs_bytes);
#else
    len                            = i2d_re_X509_tbs(certificate, &p_tbs_bytes);
    LOG_I("len = %d", len);
#endif

    /*Calculate digest for signing*/
    uint8_t digest[32] = {0};
    size_t digestLen   = sizeof(digest);
    sss_digest_t digestCtx;
    status = sss_digest_context_init(&digestCtx, &pCtx->session, kAlgorithm_SSS_SHA256, kMode_SSS_Digest);
    status = sss_digest_one_go(&digestCtx, &tbs_bytes[0], len, digest, &digestLen);
    sss_digest_context_free(&digestCtx);
    if (status != kStatus_SSS_Success)
        return status;

    status = kStatus_SSS_Fail;
    /*Sign digest*/
    sss_asymmetric_t asymmCtx;
    uint8_t signature[256];
    uint8_t *p_signature = &signature[0];
    size_t signatureLen  = sizeof(signature);
    status = sss_asymmetric_context_init(&asymmCtx, &pCtx->session, &obj, kAlgorithm_SSS_SHA256, kMode_SSS_Sign);
    status = sss_asymmetric_sign_digest(&asymmCtx, digest, digestLen, signature, &signatureLen);
    if (status != kStatus_SSS_Success)
        return status;

        /*Add signature to certificate structure*/
#if (OPENSSL_VERSION_NUMBER < 0x10100000L)
    ASN1_BIT_STRING *sig = certificate->signature;
#else
    ASN1_BIT_STRING *sig;
    const X509_ALGOR *tmp_algo = NULL;
    X509_get0_signature((const ASN1_BIT_STRING **)&sig, &tmp_algo, (const X509 *)certificate);
    X509_ALGOR *algo = (X509_ALGOR *)tmp_algo;
    X509_ALGOR_set0(algo, OBJ_nid2obj(type), V_ASN1_NULL, NULL);
#endif
    len = ASN1_BIT_STRING_set(sig, p_signature, signatureLen);

    unsigned char cert[2000];
    unsigned char *pCert = &cert[0];
    len                  = i2d_X509(certificate, NULL);
    if (len == 0) {
        LOG_I("i2d_X509 Failed");
        return kStatus_SSS_Fail;
    }
    len = i2d_X509(certificate, &pCert);
    LOG_AU8_I(cert, len);

    sss_object_t cert_obj;
    sss_key_part_t keyPart       = kSSS_KeyPart_Default;
    sss_cipher_type_t cipherType = kSSS_CipherType_Binary;
    status                       = sss_key_object_init(&cert_obj, &pCtx->ks);
    status = sss_key_object_allocate_handle(&cert_obj, keyId + 1, keyPart, cipherType, len, kKeyObject_Mode_Persistent);
    status = sss_key_store_set_key(&pCtx->ks, &cert_obj, cert, len, len * 8, NULL, 0);
    if (status != kStatus_SSS_Success)
        LOG_E("Failed to inject generated certificate");
    else
        LOG_I("Success");

    return status;
}

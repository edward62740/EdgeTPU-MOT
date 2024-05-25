var a00091 =
[
    [ "SPDIF eDMA Driver", "a00092.html", "a00092" ],
    [ "spdif_config_t", "a00091.html#a00686", [
      [ "isTxAutoSync", "a00091.html#aa62aab66a7b03f444074a6abab83294c", null ],
      [ "isRxAutoSync", "a00091.html#a8e7a4d603d3fdbca6d204c21a9729384", null ],
      [ "DPLLClkSource", "a00091.html#a51dfc55f76c63adbb2956c9be1802fab", null ],
      [ "txClkSource", "a00091.html#aae2efeac0c0a15d308758688656f21e7", null ],
      [ "rxFullSelect", "a00091.html#abad0c032f98e052695e55ba8f8dc51dd", null ],
      [ "txFullSelect", "a00091.html#a19570d0d87d133c863cea6eb7a374bad", null ],
      [ "uChannelSrc", "a00091.html#a55644590c92b55f32cccc120df5a91ba", null ],
      [ "txSource", "a00091.html#add620ef347dc5fc0b5fc4626a6d0816f", null ],
      [ "validityConfig", "a00091.html#a662ae617d9706faedff59b4e7ebfc563", null ],
      [ "gain", "a00091.html#af2eeb1f3e4a229d81b00a2e293bd5b8f", null ]
    ] ],
    [ "spdif_transfer_t", "a00091.html#a00688", [
      [ "data", "a00091.html#aafa6dcdb00a7953d3368916ec776f3f2", null ],
      [ "qdata", "a00091.html#a3f670504281a0bf4ee74d831d965b1c2", null ],
      [ "udata", "a00091.html#a547ec34f70b0c0b803a212301b0e9bbc", null ],
      [ "dataSize", "a00091.html#ae44eb4e4b3141f9478e4b35a9bae8af3", null ]
    ] ],
    [ "spdif_handle_t", "a00091.html#a00411", [
      [ "state", "a00091.html#a530888b8eb6d5b6383ac803ef5687cac", null ],
      [ "callback", "a00091.html#ae6a348888f71f09a0a2167064b627fef", null ],
      [ "userData", "a00091.html#a453888e826bebbf6d3cefbc1d5c37aa2", null ],
      [ "spdifQueue", "a00091.html#acfad92f425b01b23f503cc6f6cbf0199", null ],
      [ "transferSize", "a00091.html#a7ba10e18fe4300506084981f16ba7376", null ],
      [ "queueUser", "a00091.html#a8f24af404b79c5b742fc54aa78c4dc06", null ],
      [ "queueDriver", "a00091.html#ac3925e65ffb3d41a9c637001844f6560", null ],
      [ "watermark", "a00091.html#a286a39db68bb63e8d64844dc94cb6b7a", null ]
    ] ],
    [ "FSL_SPDIF_DRIVER_VERSION", "a00091.html#ga036bac4f2d40e143c5528a10038767c1", null ],
    [ "SPDIF_XFER_QUEUE_SIZE", "a00091.html#ga3e78a3155b1291a9fa00d0461d6e490c", null ],
    [ "spdif_transfer_callback_t", "a00091.html#ga8749a36136a9b55600479d61b55596ec", [
      [ "kStatus_SPDIF_RxDPLLLocked", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a0489bbc59ac12902e0cf3c134e72363e", null ],
      [ "kStatus_SPDIF_TxFIFOError", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a278a5b9459f7c6a06b0026d00ecc63cd", null ],
      [ "kStatus_SPDIF_TxFIFOResync", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a051d7f8261885f910a4f04c4a7a8f410", null ],
      [ "kStatus_SPDIF_RxCnew", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a259512eb9a030bbc2188b71008058808", null ],
      [ "kStatus_SPDIF_ValidatyNoGood", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a6da0731f8caeb6cc08bb5f8190d01f30", null ],
      [ "kStatus_SPDIF_RxIllegalSymbol", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a2da36cd8a86bdfa8d9b4fead3dd543ad", null ],
      [ "kStatus_SPDIF_RxParityBitError", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058afe0341b5aea2f8946c5a5a102e676483", null ],
      [ "kStatus_SPDIF_UChannelOverrun", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a02d356ebee96b06bd87d13551d09e73f", null ],
      [ "kStatus_SPDIF_QChannelOverrun", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a07b3373290d7d037ff3a859132967d53", null ],
      [ "kStatus_SPDIF_UQChannelSync", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a3a6c65eabeb242b3910327da198c87d3", null ],
      [ "kStatus_SPDIF_UQChannelFrameError", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058af44eacef3e48bd00a1fd5e08c550f9cb", null ],
      [ "kStatus_SPDIF_RxFIFOError", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a2360fecc1ffe49c63f8dd5b64905a33a", null ],
      [ "kStatus_SPDIF_RxFIFOResync", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058ab206b050696acf5758e8b6c0bdf5b6fa", null ],
      [ "kStatus_SPDIF_LockLoss", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a9390ad9ad5ce85f7224c759604f5a194", null ],
      [ "kStatus_SPDIF_TxIdle", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a646939956dd08768873e67af0e2690fc", null ],
      [ "kStatus_SPDIF_RxIdle", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058ac458cedc1f2f208c0e04e3f532dc0595", null ],
      [ "kStatus_SPDIF_QueueFull", "a00091.html#gga0ae1e3bf78c960c83e2d437efd802058a700d25ed1454e469f32de0ade5a78e87", null ]
    ] ],
    [ "spdif_rxfull_select_t", "a00091.html#ga85d08a5eba9eaf262597f4f3dfad19a0", [
      [ "kSPDIF_RxFull1Sample", "a00091.html#gga85d08a5eba9eaf262597f4f3dfad19a0ad6a45c3b1bb3397ce7dd2e5c9af8c1c9", null ],
      [ "kSPDIF_RxFull4Samples", "a00091.html#gga85d08a5eba9eaf262597f4f3dfad19a0a84f7309ca36974eefa1a2d970eba7995", null ],
      [ "kSPDIF_RxFull8Samples", "a00091.html#gga85d08a5eba9eaf262597f4f3dfad19a0a71af34697ac17ca08eb26d7c4d6dd27f", null ],
      [ "kSPDIF_RxFull16Samples", "a00091.html#gga85d08a5eba9eaf262597f4f3dfad19a0ab585922a28236a3e9b8cffbdd3793cee", null ]
    ] ],
    [ "spdif_txempty_select_t", "a00091.html#gad58b90afceef70798b8be683f3017c71", [
      [ "kSPDIF_TxEmpty0Sample", "a00091.html#ggad58b90afceef70798b8be683f3017c71a835efeacfa1b7bc07a723c3683407b4e", null ],
      [ "kSPDIF_TxEmpty4Samples", "a00091.html#ggad58b90afceef70798b8be683f3017c71a38d0ed7a9ff015721f0edcd71d696e18", null ],
      [ "kSPDIF_TxEmpty8Samples", "a00091.html#ggad58b90afceef70798b8be683f3017c71afbd58fcc26965a9b45865764d775284f", null ],
      [ "kSPDIF_TxEmpty12Samples", "a00091.html#ggad58b90afceef70798b8be683f3017c71a9c74437e11128b87fbd5bb644c4790e4", null ]
    ] ],
    [ "spdif_uchannel_source_t", "a00091.html#ga01edc35977dbab89a7201285c61c4a3e", [
      [ "kSPDIF_NoUChannel", "a00091.html#gga01edc35977dbab89a7201285c61c4a3eaf159efd3316b59e34b7288448c4b8c4f", null ],
      [ "kSPDIF_UChannelFromRx", "a00091.html#gga01edc35977dbab89a7201285c61c4a3eaec4bf6dbe6a1346b33c59c392ea1fbe6", null ],
      [ "kSPDIF_UChannelFromTx", "a00091.html#gga01edc35977dbab89a7201285c61c4a3ea65a3b2cbcb558fdc5870f009d9d6bd80", null ]
    ] ],
    [ "spdif_gain_select_t", "a00091.html#gab5381bc40ad0b86e9453495598911e2c", [
      [ "kSPDIF_GAIN_24", "a00091.html#ggab5381bc40ad0b86e9453495598911e2cac438b3169eb2675d11540a839e99f502", null ],
      [ "kSPDIF_GAIN_16", "a00091.html#ggab5381bc40ad0b86e9453495598911e2caa776465b4de5bf2fa610f1c2366c063f", null ],
      [ "kSPDIF_GAIN_12", "a00091.html#ggab5381bc40ad0b86e9453495598911e2cad54dc0e7908199d3fe24ef53d608b544", null ],
      [ "kSPDIF_GAIN_8", "a00091.html#ggab5381bc40ad0b86e9453495598911e2caafee91827c9ea238ad6d6478718e98a0", null ],
      [ "kSPDIF_GAIN_6", "a00091.html#ggab5381bc40ad0b86e9453495598911e2ca16500e129c2451b60faf4a3f4464df43", null ],
      [ "kSPDIF_GAIN_4", "a00091.html#ggab5381bc40ad0b86e9453495598911e2ca84750080b7f27da810fe19abe64f776f", null ],
      [ "kSPDIF_GAIN_3", "a00091.html#ggab5381bc40ad0b86e9453495598911e2ca1008286af785dcb2dde9131765217980", null ]
    ] ],
    [ "spdif_tx_source_t", "a00091.html#ga2847432f45adbafdb2f9f7e5c25d9b2e", [
      [ "kSPDIF_txFromReceiver", "a00091.html#gga2847432f45adbafdb2f9f7e5c25d9b2eaffd9df4017e5af840e5c34c050791f02", null ],
      [ "kSPDIF_txNormal", "a00091.html#gga2847432f45adbafdb2f9f7e5c25d9b2ea1e6073994dc81cb60c229a8ce166222f", null ]
    ] ],
    [ "spdif_validity_config_t", "a00091.html#gad5615c4913a612ee1dc67e357e905f2c", [
      [ "kSPDIF_validityFlagAlwaysSet", "a00091.html#ggad5615c4913a612ee1dc67e357e905f2ca9cf3c378520222b1ffe3e584cae0ac67", null ],
      [ "kSPDIF_validityFlagAlwaysClear", "a00091.html#ggad5615c4913a612ee1dc67e357e905f2ca96145e43a2a36d2c7dcd6ce5242743b1", null ],
      [ "kSPDIF_RxDPLLLocked", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dadab8774a200c36eccf9d7b439ba21d1ac8", null ],
      [ "kSPDIF_TxFIFOError", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada74d3b2d4e4ee35eed58f4d30c6d2709b", null ],
      [ "kSPDIF_TxFIFOResync", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dadaf40104f71f214bb1c0a6a2c0fdcdaf36", null ],
      [ "kSPDIF_RxControlChannelChange", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada5917028824aefbf16086b3021e414340", null ],
      [ "kSPDIF_ValidityFlagNoGood", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada7de7b2231ecb6e20e43d0494fe92e63c", null ],
      [ "kSPDIF_RxIllegalSymbol", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dadaa71c8ef82de54ffac949e31010d031de", null ],
      [ "kSPDIF_RxParityBitError", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada1838e42bf8376a18762297b6aa3c4472", null ],
      [ "kSPDIF_UChannelReceiveRegisterFull", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dadad228d31963d41a20e3567ce6f633acc6", null ],
      [ "kSPDIF_UChannelReceiveRegisterOverrun", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada3b7dc66169552ca7798d686d68de6b7d", null ],
      [ "kSPDIF_QChannelReceiveRegisterFull", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada80ac61a0edef14ba03035176ab326331", null ],
      [ "kSPDIF_QChannelReceiveRegisterOverrun", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada64e59711062fadb98877919d7c5887e9", null ],
      [ "kSPDIF_UQChannelSync", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada6e94d542bd83c5bfc043fdbc44fcd43b", null ],
      [ "kSPDIF_UQChannelFrameError", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dadac07a719f553a9e41bf3e082a5beaf107", null ],
      [ "kSPDIF_RxFIFOError", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada1a27bda77a59e1db5a1913bb8971281a", null ],
      [ "kSPDIF_RxFIFOResync", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dadad2e29bb00f59f56babeebd1f0e73726a", null ],
      [ "kSPDIF_LockLoss", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada1f134eebfeb0740e56c13c9b45fb8434", null ],
      [ "kSPDIF_TxFIFOEmpty", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada82af9fd1290d8b3b63420b8d92257819", null ],
      [ "kSPDIF_RxFIFOFull", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dada9797531448d68e40b3befb91810bd09c", null ],
      [ "kSPDIF_AllInterrupt", "a00091.html#gga01aea4eb01aa6415eee118b5a5ee3dadaf55570f1f8a886f06d64ed2bfc0650a4", null ],
      [ "kSPDIF_RxDMAEnable", "a00091.html#gga7646ae03981912f97626c39dfe9318daa19fc49573180ffcc3df5f9a27c797209", null ],
      [ "kSPDIF_TxDMAEnable", "a00091.html#gga7646ae03981912f97626c39dfe9318daad508f6bf16fd6349b03fc8f3b38d06e8", null ]
    ] ],
    [ "SPDIF_Init", "a00091.html#ga6b4e9ed2c903d62af53a8d3887126413", null ],
    [ "SPDIF_GetDefaultConfig", "a00091.html#ga11b262ea3923fd80f78b959d9718e00b", null ],
    [ "SPDIF_Deinit", "a00091.html#ga2631baffebf03ec2a587580112dd56f3", null ],
    [ "SPDIF_GetInstance", "a00091.html#gacbd5fc4df43e1a249493b5324f94e54d", null ],
    [ "SPDIF_TxFIFOReset", "a00091.html#ga85e3497baec03b34f728541154ccc31d", null ],
    [ "SPDIF_RxFIFOReset", "a00091.html#gaa74cc12474982eb7c46adaa7d4891626", null ],
    [ "SPDIF_TxEnable", "a00091.html#ga5b76be7fc7148dec7d4493398103a94d", null ],
    [ "SPDIF_RxEnable", "a00091.html#gad705a47a3b61bb334125e54fc70ed2ae", null ],
    [ "SPDIF_GetStatusFlag", "a00091.html#gafe28766cd0629d77c2ec2f5986155ee7", null ],
    [ "SPDIF_ClearStatusFlags", "a00091.html#ga39bddc0768ce21bbbd6d3eae837b206e", null ],
    [ "SPDIF_EnableInterrupts", "a00091.html#ga487f880569d931ee08c74a83332862ae", null ],
    [ "SPDIF_DisableInterrupts", "a00091.html#ga0c0c7200763825eba3f10bbe7d3439b8", null ],
    [ "SPDIF_EnableDMA", "a00091.html#ga1d4992842b29cef3c71ad5b5fbd664d3", null ],
    [ "SPDIF_TxGetLeftDataRegisterAddress", "a00091.html#ga72dc7737eb0911c1c887d53cf9e1c8ed", null ],
    [ "SPDIF_TxGetRightDataRegisterAddress", "a00091.html#gad6fe7d183782a4d078e0cd182d5859a3", null ],
    [ "SPDIF_RxGetLeftDataRegisterAddress", "a00091.html#gabd6b7be3a3b545c6f6b2d35f72cadef4", null ],
    [ "SPDIF_RxGetRightDataRegisterAddress", "a00091.html#ga4485709f8d45c96f5dce7dd85ecef07f", null ],
    [ "SPDIF_TxSetSampleRate", "a00091.html#gad49d52850fb379566953c66bf1f93a54", null ],
    [ "SPDIF_GetRxSampleRate", "a00091.html#gaf119e024d14a0c288b743dd17bbef687", null ],
    [ "SPDIF_WriteBlocking", "a00091.html#gab0eb427edd9cc4e5ece878563b9a6a8c", null ],
    [ "SPDIF_WriteLeftData", "a00091.html#ga2b78216d0f4af76d8b9ff82101fc8adf", null ],
    [ "SPDIF_WriteRightData", "a00091.html#ga48edd7722d5a59091e6da7756ef25f0a", null ],
    [ "SPDIF_WriteChannelStatusHigh", "a00091.html#ga3804918ec78e961fb33b12094bf732e8", null ],
    [ "SPDIF_WriteChannelStatusLow", "a00091.html#ga6fd048c19cff5ef48c133ca0ffbe23c1", null ],
    [ "SPDIF_ReadBlocking", "a00091.html#ga2fb9e8f8bafa60358626840d0442265e", null ],
    [ "SPDIF_ReadLeftData", "a00091.html#ga49e6ea71b76ff7e3e97da8ffd8cd83ea", null ],
    [ "SPDIF_ReadRightData", "a00091.html#ga34ebdb7cd9a243303bc21cd9b16f82c3", null ],
    [ "SPDIF_ReadChannelStatusHigh", "a00091.html#ga6897be2df12c8df71d4f6bff5c7546c1", null ],
    [ "SPDIF_ReadChannelStatusLow", "a00091.html#ga23d1a99377b7f4b26c25a79ece8231ad", null ],
    [ "SPDIF_ReadQChannel", "a00091.html#ga7a5d474af2b6b256df0bf650dd8154c0", null ],
    [ "SPDIF_ReadUChannel", "a00091.html#gac0f070017fc4e1db6b0ceeef990c3f7e", null ],
    [ "SPDIF_TransferTxCreateHandle", "a00091.html#ga2e4eb257e2da537eb2d47a31ea47fbb6", null ],
    [ "SPDIF_TransferRxCreateHandle", "a00091.html#gac44c0edc9e22798fb840f38f1c3749ba", null ],
    [ "SPDIF_TransferSendNonBlocking", "a00091.html#ga155e14a0591e50af578cb8309171c112", null ],
    [ "SPDIF_TransferReceiveNonBlocking", "a00091.html#ga04fe7c5e2cf560687b07f5de908da255", null ],
    [ "SPDIF_TransferGetSendCount", "a00091.html#ga56e6fe5a91de5d1f38896128c4958a94", null ],
    [ "SPDIF_TransferGetReceiveCount", "a00091.html#gaad53ad9d352550e8fa84a3200ee977c9", null ],
    [ "SPDIF_TransferAbortSend", "a00091.html#ga004b052fdaa3fd78dd7bafc721b28f0c", null ],
    [ "SPDIF_TransferAbortReceive", "a00091.html#ga5dd15b214687cf6a120142b0152ac18c", null ],
    [ "SPDIF_TransferTxHandleIRQ", "a00091.html#ga14adaa4280f963a9e6c0b6d760173182", null ],
    [ "SPDIF_TransferRxHandleIRQ", "a00091.html#gaa1ab4a1d05d2e708ab9896f0131506b4", null ]
];
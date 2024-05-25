/** @file wlan_txpwrlimit_cfg_murata_1ZM_CA
 *
 *  @brief  This file provides Murata 1ZM WLAN CA Tx Power Limit APIs.
 *
 *  Copyright 2008-2020 NXP
 *
 *  NXP CONFIDENTIAL
 *  The source code contained or described herein and all documents related to
 *  the source code ("Materials") are owned by NXP, its
 *  suppliers and/or its licensors. Title to the Materials remains with NXP,
 *  its suppliers and/or its licensors. The Materials contain
 *  trade secrets and proprietary and confidential information of NXP, its
 *  suppliers and/or its licensors. The Materials are protected by worldwide copyright
 *  and trade secret laws and treaty provisions. No part of the Materials may be
 *  used, copied, reproduced, modified, published, uploaded, posted,
 *  transmitted, distributed, or disclosed in any way without NXP's prior
 *  express written permission.
 *
 *  No license under any patent, copyright, trade secret or other intellectual
 *  property right is granted to or conferred upon you by disclosure or delivery
 *  of the Materials, either expressly, by implication, inducement, estoppel or
 *  otherwise. Any license under such intellectual property rights must be
 *  express and approved by NXP in writing.
 *
 */

static wlan_chanlist_t chanlist_2g_cfg = {
    .num_chans = 11,
    .chan_info[0] =
        {
            .chan_num                     = 1,
            .chan_freq                    = 2412,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[1] =
        {
            .chan_num                     = 2,
            .chan_freq                    = 2417,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[2] =
        {
            .chan_num                     = 3,
            .chan_freq                    = 2422,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[3] =
        {
            .chan_num                     = 4,
            .chan_freq                    = 2427,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[4] =
        {
            .chan_num                     = 5,
            .chan_freq                    = 2432,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[5] =
        {
            .chan_num                     = 6,
            .chan_freq                    = 2437,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[6] =
        {
            .chan_num                     = 7,
            .chan_freq                    = 2442,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[7] =
        {
            .chan_num                     = 8,
            .chan_freq                    = 2447,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[8] =
        {
            .chan_num                     = 9,
            .chan_freq                    = 2452,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[9] =
        {
            .chan_num                     = 10,
            .chan_freq                    = 2457,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[10] =
        {
            .chan_num                     = 11,
            .chan_freq                    = 2462,
            .passive_scan_or_radar_detect = false,
        },
};

#ifdef CONFIG_5GHz_SUPPORT
static wlan_chanlist_t chanlist_5g_cfg = {
    .num_chans = 22,
    .chan_info[0] =
        {
            .chan_num                     = 36,
            .chan_freq                    = 5180,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[1] =
        {
            .chan_num                     = 40,
            .chan_freq                    = 5200,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[2] =
        {
            .chan_num                     = 44,
            .chan_freq                    = 5220,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[3] =
        {
            .chan_num                     = 48,
            .chan_freq                    = 5240,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[4] =
        {
            .chan_num                     = 52,
            .chan_freq                    = 5260,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[5] =
        {
            .chan_num                     = 56,
            .chan_freq                    = 5280,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[6] =
        {
            .chan_num                     = 60,
            .chan_freq                    = 5300,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[7] =
        {
            .chan_num                     = 64,
            .chan_freq                    = 5320,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[8] =
        {
            .chan_num                     = 100,
            .chan_freq                    = 5500,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[9] =
        {
            .chan_num                     = 104,
            .chan_freq                    = 5520,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[10] =
        {
            .chan_num                     = 108,
            .chan_freq                    = 5540,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[11] =
        {
            .chan_num                     = 112,
            .chan_freq                    = 5560,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[12] =
        {
            .chan_num                     = 116,
            .chan_freq                    = 5580,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[13] =
        {
            .chan_num                     = 132,
            .chan_freq                    = 5660,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[14] =
        {
            .chan_num                     = 136,
            .chan_freq                    = 5680,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[15] =
        {
            .chan_num                     = 140,
            .chan_freq                    = 5700,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[16] =
        {
            .chan_num                     = 144,
            .chan_freq                    = 5720,
            .passive_scan_or_radar_detect = true,
        },
    .chan_info[17] =
        {
            .chan_num                     = 149,
            .chan_freq                    = 5745,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[18] =
        {
            .chan_num                     = 153,
            .chan_freq                    = 5765,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[19] =
        {
            .chan_num                     = 157,
            .chan_freq                    = 5785,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[20] =
        {
            .chan_num                     = 161,
            .chan_freq                    = 5805,
            .passive_scan_or_radar_detect = false,
        },
    .chan_info[21] =
        {
            .chan_num                     = 165,
            .chan_freq                    = 5825,
            .passive_scan_or_radar_detect = false,
        },
};
#endif

#ifndef CONFIG_11AC
static wifi_txpwrlimit_t tx_pwrlimit_2g_cfg = {
    .subband   = (wifi_SubBand_t)0x00,
    .num_chans = 14,
    .txpwrlimit_config[0] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 1,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[1] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 2,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[2] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 3,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[3] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 4,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 17}, {2, 16}, {3, 16}, {4, 16}, {5, 15}, {6, 15}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[4] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 5,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 17}, {2, 16}, {3, 16}, {4, 16}, {5, 15}, {6, 15}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[5] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 6,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 17}, {2, 16}, {3, 16}, {4, 16}, {5, 15}, {6, 15}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[6] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 7,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 17}, {2, 16}, {3, 16}, {4, 16}, {5, 15}, {6, 15}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[7] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 8,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 17}, {2, 16}, {3, 16}, {4, 16}, {5, 15}, {6, 15}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[8] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 9,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[9] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 10,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[10] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 11,
                },
            .txpwrlimit_entry = {{0, 17}, {1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[11] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 12,
                },
            .txpwrlimit_entry = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[12] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2407,
                    .chan_width = 20,
                    .chan_num   = 13,
                },
            .txpwrlimit_entry = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}},
        },
    .txpwrlimit_config[13] =
        {
            .num_mod_grps = 10,
            .chan_desc =
                {
                    .start_freq = 2414,
                    .chan_width = 20,
                    .chan_num   = 14,
                },
            .txpwrlimit_entry = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}},
        },
};

#ifdef CONFIG_5GHz_SUPPORT
static wifi_txpwrlimit_t tx_pwrlimit_5g_cfg =
    {
        .subband   = (wifi_SubBand_t)0x00,
        .num_chans = 39,
        .txpwrlimit_config[0] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 36,
                    },
                .txpwrlimit_entry = {{1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 12}, {8, 12}, {9, 12}},
            },
        .txpwrlimit_config[1] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 40,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 12}, {8, 12}, {9, 12}},
            },
        .txpwrlimit_config[2] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 44,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[3] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 48,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[4] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 52,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[5] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 56,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[6] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 60,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 12}, {8, 12}, {9, 12}},
            },
        .txpwrlimit_config[7] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 64,
                    },
                .txpwrlimit_entry = {{1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 12}, {8, 12}, {9, 12}},
            },
        .txpwrlimit_config[8] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 100,
                    },
                .txpwrlimit_entry = {{1, 14}, {2, 14}, {3, 14}, {4, 13}, {5, 13}, {6, 13}, {7, 12}, {8, 12}, {9, 12}},
            },
        .txpwrlimit_config[9] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 104,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 12}, {8, 12}, {9, 12}},
            },
        .txpwrlimit_config[10] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 108,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[11] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 112,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[12] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 116,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[13] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 120,
                    },
                .txpwrlimit_entry = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}},
            },
        .txpwrlimit_config[14] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 124,
                    },
                .txpwrlimit_entry = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}},
            },
        .txpwrlimit_config[15] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 128,
                    },
                .txpwrlimit_entry = {{1, 0}, {2, 0}, {3, 0}, {4, 0}, {5, 0}, {6, 0}, {7, 0}, {8, 0}, {9, 0}},
            },
        .txpwrlimit_config[16] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 132,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[17] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 136,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[18] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 140,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[19] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 144,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[20] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 149,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[21] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 153,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[22] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 157,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[23] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 161,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[24] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 165,
                    },
                .txpwrlimit_entry = {{1, 15}, {2, 15}, {3, 15}, {4, 14}, {5, 14}, {6, 14}, {7, 14}, {8, 14}, {9, 14}},
            },
        .txpwrlimit_config[25] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 183,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[26] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 184,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[27] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 185,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[28] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 187,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[29] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 188,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[30] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 189,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[31] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 192,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[32] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 196,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[33] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 7,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[34] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 8,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[35] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 11,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[36] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 12,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[37] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 16,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
        .txpwrlimit_config[38] =
            {
                .num_mod_grps = 9,
                .chan_desc =
                    {
                        .start_freq = 5000,
                        .chan_width = 20,
                        .chan_num   = 34,
                    },
                .txpwrlimit_entry = {{1, 17}, {2, 16}, {3, 14}, {4, 17}, {5, 16}, {6, 14}, {7, 17}, {8, 16}, {9, 14}},
            },
};
#endif
#else
static wifi_txpwrlimit_t
    tx_pwrlimit_2g_cfg =
        {
            .subband   = (wifi_SubBand_t)0x00,
            .num_chans = 14,
            .txpwrlimit_config[0] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 1,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 14},
                                         {2, 14},
                                         {3, 14},
                                         {4, 13},
                                         {5, 13},
                                         {6, 13},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[1] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 2,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 14},
                                         {2, 14},
                                         {3, 14},
                                         {4, 13},
                                         {5, 13},
                                         {6, 13},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[2] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 3,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 14},
                                         {2, 14},
                                         {3, 14},
                                         {4, 13},
                                         {5, 13},
                                         {6, 13},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[3] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 4,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 17},
                                         {2, 16},
                                         {3, 16},
                                         {4, 16},
                                         {5, 15},
                                         {6, 15},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[4] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 5,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 17},
                                         {2, 16},
                                         {3, 16},
                                         {4, 16},
                                         {5, 15},
                                         {6, 15},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[5] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 6,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 17},
                                         {2, 16},
                                         {3, 16},
                                         {4, 16},
                                         {5, 15},
                                         {6, 15},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[6] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 7,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 17},
                                         {2, 16},
                                         {3, 16},
                                         {4, 16},
                                         {5, 15},
                                         {6, 15},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[7] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 8,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 17},
                                         {2, 16},
                                         {3, 16},
                                         {4, 16},
                                         {5, 15},
                                         {6, 15},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[8] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 9,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 14},
                                         {2, 14},
                                         {3, 14},
                                         {4, 13},
                                         {5, 13},
                                         {6, 13},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[9] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 10,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 14},
                                         {2, 14},
                                         {3, 14},
                                         {4, 13},
                                         {5, 13},
                                         {6, 13},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[10] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 11,
                        },
                    .txpwrlimit_entry = {{0, 17},
                                         {1, 14},
                                         {2, 14},
                                         {3, 14},
                                         {4, 13},
                                         {5, 13},
                                         {6, 13},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[11] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 12,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 0},
                                         {2, 0},
                                         {3, 0},
                                         {4, 0},
                                         {5, 0},
                                         {6, 0},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[12] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2407,
                            .chan_width = 20,
                            .chan_num   = 13,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 0},
                                         {2, 0},
                                         {3, 0},
                                         {4, 0},
                                         {5, 0},
                                         {6, 0},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
            .txpwrlimit_config[13] =
                {
                    .num_mod_grps = 12,
                    .chan_desc =
                        {
                            .start_freq = 2414,
                            .chan_width = 20,
                            .chan_num   = 14,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 0},
                                         {2, 0},
                                         {3, 0},
                                         {4, 0},
                                         {5, 0},
                                         {6, 0},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0}},
                },
};

#ifdef CONFIG_5GHz_SUPPORT
static wifi_txpwrlimit_t
    tx_pwrlimit_5g_cfg =
        {
            .subband   = (wifi_SubBand_t)0x00,
            .num_chans = 39,
            .txpwrlimit_config[0] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 36,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 14},
                                         {2, 14},
                                         {3, 14},
                                         {4, 13},
                                         {5, 13},
                                         {6, 13},
                                         {7, 12},
                                         {8, 12},
                                         {9, 12},
                                         {10, 13},
                                         {11, 12},
                                         {12, 10},
                                         {13, 10},
                                         {14, 10},
                                         {15, 10}},
                },
            .txpwrlimit_config[1] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 40,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 15},
                                         {2, 15},
                                         {3, 15},
                                         {4, 14},
                                         {5, 14},
                                         {6, 14},
                                         {7, 12},
                                         {8, 12},
                                         {9, 12},
                                         {10, 14},
                                         {11, 12},
                                         {12, 10},
                                         {13, 10},
                                         {14, 10},
                                         {15, 10}},
                },
            .txpwrlimit_config[2] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 44,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 15},
                                         {2, 15},
                                         {3, 15},
                                         {4, 14},
                                         {5, 14},
                                         {6, 14},
                                         {7, 14},
                                         {8, 14},
                                         {9, 14},
                                         {10, 14},
                                         {11, 13},
                                         {12, 10},
                                         {13, 10},
                                         {14, 10},
                                         {15, 10}},
                },
            .txpwrlimit_config[3] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 48,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 15},
                                         {2, 15},
                                         {3, 15},
                                         {4, 14},
                                         {5, 14},
                                         {6, 14},
                                         {7, 14},
                                         {8, 14},
                                         {9, 14},
                                         {10, 14},
                                         {11, 13},
                                         {12, 10},
                                         {13, 10},
                                         {14, 10},
                                         {15, 10}},
                },
            .txpwrlimit_config[4] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 52,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 15},
                                         {2, 15},
                                         {3, 15},
                                         {4, 14},
                                         {5, 14},
                                         {6, 14},
                                         {7, 14},
                                         {8, 14},
                                         {9, 14},
                                         {10, 14},
                                         {11, 13},
                                         {12, 10},
                                         {13, 10},
                                         {14, 10},
                                         {15, 10}},
                },
            .txpwrlimit_config[5] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 56,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 15},
                                         {2, 15},
                                         {3, 15},
                                         {4, 14},
                                         {5, 14},
                                         {6, 14},
                                         {7, 14},
                                         {8, 14},
                                         {9, 14},
                                         {10, 14},
                                         {11, 13},
                                         {12, 10},
                                         {13, 10},
                                         {14, 10},
                                         {15, 10}},
                },
            .txpwrlimit_config[6] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 60,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 15},
                                         {2, 15},
                                         {3, 15},
                                         {4, 14},
                                         {5, 14},
                                         {6, 14},
                                         {7, 12},
                                         {8, 12},
                                         {9, 12},
                                         {10, 14},
                                         {11, 12},
                                         {12, 10},
                                         {13, 10},
                                         {14, 10},
                                         {15, 10}},
                },
            .txpwrlimit_config[7] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 64,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 14},
                         {2, 14},
                         {3, 14},
                         {4, 13},
                         {5, 13},
                         {6, 13},
                         {7, 12},
                         {8, 12},
                         {9, 12},
                         {10, 13},
                         {11, 12},
                         {12, 10},
                         {13, 10},
                         {14, 10},
                         {15, 10}},
                },
            .txpwrlimit_config[8] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 100,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 14},
                         {2, 14},
                         {3, 14},
                         {4, 13},
                         {5, 13},
                         {6, 13},
                         {7, 12},
                         {8, 12},
                         {9, 12},
                         {10, 13},
                         {11, 12},
                         {12, 10},
                         {13, 10},
                         {14, 10},
                         {15, 10}},
                },
            .txpwrlimit_config[9] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 104,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 12},
                         {8, 12},
                         {9, 12},
                         {10, 14},
                         {11, 12},
                         {12, 10},
                         {13, 10},
                         {14, 10},
                         {15, 10}},
                },
            .txpwrlimit_config[10] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 108,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 10},
                         {13, 10},
                         {14, 10},
                         {15, 10}},
                },
            .txpwrlimit_config[11] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 112,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 10},
                         {13, 10},
                         {14, 10},
                         {15, 10}},
                },
            .txpwrlimit_config[12] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 116,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[13] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 120,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 0},
                                         {2, 0},
                                         {3, 0},
                                         {4, 0},
                                         {5, 0},
                                         {6, 0},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0},
                                         {12, 0},
                                         {13, 0},
                                         {14, 0},
                                         {15, 0}},
                },
            .txpwrlimit_config[14] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 124,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 0},
                                         {2, 0},
                                         {3, 0},
                                         {4, 0},
                                         {5, 0},
                                         {6, 0},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0},
                                         {12, 0},
                                         {13, 0},
                                         {14, 0},
                                         {15, 0}},
                },
            .txpwrlimit_config[15] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 128,
                        },
                    .txpwrlimit_entry = {{0, 0},
                                         {1, 0},
                                         {2, 0},
                                         {3, 0},
                                         {4, 0},
                                         {5, 0},
                                         {6, 0},
                                         {7, 0},
                                         {8, 0},
                                         {9, 0},
                                         {10, 0},
                                         {11, 0},
                                         {12, 0},
                                         {13, 0},
                                         {14, 0},
                                         {15, 0}},
                },
            .txpwrlimit_config[16] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 132,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[17] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 136,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[18] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 140,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[19] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 144,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[20] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 149,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[21] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 153,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[22] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 157,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[23] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 161,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[24] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 165,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 15},
                         {2, 15},
                         {3, 15},
                         {4, 14},
                         {5, 14},
                         {6, 14},
                         {7, 14},
                         {8, 14},
                         {9, 14},
                         {10, 14},
                         {11, 13},
                         {12, 14},
                         {13, 13},
                         {14, 13},
                         {15, 13}},
                },
            .txpwrlimit_config[25] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 183,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[26] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 184,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[27] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 185,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[28] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 187,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[29] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 188,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[30] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 189,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[31] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 192,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[32] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 196,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[33] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 7,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[34] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 8,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[35] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 11,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[36] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 12,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[37] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 16,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
            .txpwrlimit_config[38] =
                {
                    .num_mod_grps = 16,
                    .chan_desc =
                        {
                            .start_freq = 5000,
                            .chan_width = 20,
                            .chan_num   = 34,
                        },
                    .txpwrlimit_entry =
                        {{0, 0},
                         {1, 17},
                         {2, 16},
                         {3, 14},
                         {4, 17},
                         {5, 16},
                         {6, 14},
                         {7, 17},
                         {8, 16},
                         {9, 14},
                         {10, 15},
                         {11, 14},
                         {12, 15},
                         {13, 15},
                         {14, 14},
                         {15, 13}},
                },
};
#endif /* CONFIG_5GHz_SUPPORT */
#endif /* CONFIG_11AC */

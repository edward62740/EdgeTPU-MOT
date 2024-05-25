/*
 * Copyright (c) 2015 - 2016, Freescale Semiconductor, Inc.
 * Copyright 2016 - 2017 NXP
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __USB_DEVICE_DESCRIPTOR_H__
#define __USB_DEVICE_DESCRIPTOR_H__

/*******************************************************************************
 * Definitions
 ******************************************************************************/

#define USB_DEVICE_SPECIFIC_BCD_VERSION (0x0200U)
#define USB_DEVICE_DEMO_BCD_VERSION     (0x0101U)

#define USB_DEVICE_VID (0x1FC9U)
#define USB_DEVICE_PID (0x00A2U)

#define USB_DEVICE_CLASS    (0x00U)
#define USB_DEVICE_SUBCLASS (0x00U)
#define USB_DEVICE_PROTOCOL (0x00U)

#define USB_DEVICE_MAX_POWER (0x32U)

#define USB_DESCRIPTOR_LENGTH_CONFIGURATION_ALL  (sizeof(g_UsbDeviceConfigurationDescriptor))
#define USB_DESCRIPTOR_LENGTH_HID_GENERIC_REPORT (sizeof(g_UsbDeviceHidGenericReportDescriptor))
#define USB_DESCRIPTOR_LENGTH_HID                (9U)
#define USB_DESCRIPTOR_LENGTH_STRING0            (sizeof(g_UsbDeviceString0))
#define USB_DESCRIPTOR_LENGTH_STRING1            (sizeof(g_UsbDeviceString1))
#define USB_DESCRIPTOR_LENGTH_STRING2            (sizeof(g_UsbDeviceString2))

#define USB_DEVICE_CONFIGURATION_COUNT (1U)
#if (defined(USB_DEVICE_CONFIG_ROOT2_TEST) && (USB_DEVICE_CONFIG_ROOT2_TEST > 0U))
#define USB_DEVICE_STRING_COUNT (4U)
#else
#define USB_DEVICE_STRING_COUNT (3U)
#endif
#define USB_DEVICE_LANGUAGE_COUNT (1U)

#define USB_HID_GENERIC_CONFIGURE_INDEX (1U)
#define USB_HID_GENERIC_INTERFACE_COUNT (1U)
/*IN lenght is same with out lenght, in case in length is not equal to out length, pleae check*/
/*USB_DeviceHidSend/Recv to make sure the length parameter is right*/
#define USB_HID_GENERIC_IN_BUFFER_LENGTH  (8U)
#define USB_HID_GENERIC_OUT_BUFFER_LENGTH (8U)
#define USB_HID_GENERIC_ENDPOINT_COUNT    (2U)
#define USB_HID_GENERIC_INTERFACE_INDEX   (0U)
#define USB_HID_GENERIC_ENDPOINT_IN       (1U)
#define USB_HID_GENERIC_ENDPOINT_OUT      (2U)

#define USB_HID_GENERIC_INTERFACE_ALTERNATE_COUNT (1U)
#define USB_HID_GENERIC_INTERFACE_ALTERNATE_0     (0U)

#define USB_HID_GENERIC_CLASS    (0x03U)
#define USB_HID_GENERIC_SUBCLASS (0x00U)
#define USB_HID_GENERIC_PROTOCOL (0x00U)

#define HS_HID_GENERIC_INTERRUPT_OUT_PACKET_SIZE (8U)
#define FS_HID_GENERIC_INTERRUPT_OUT_PACKET_SIZE (8U)
#define HS_HID_GENERIC_INTERRUPT_OUT_INTERVAL    (0x04U) /* 2^(4-1) = 1ms */
#define FS_HID_GENERIC_INTERRUPT_OUT_INTERVAL    (0x01U)

#define HS_HID_GENERIC_INTERRUPT_IN_PACKET_SIZE (8U)
#define FS_HID_GENERIC_INTERRUPT_IN_PACKET_SIZE (8U)
#define HS_HID_GENERIC_INTERRUPT_IN_INTERVAL    (0x04U) /* 2^(4-1) = 1ms */
#define FS_HID_GENERIC_INTERRUPT_IN_INTERVAL    (0x01U)

/*******************************************************************************
 * API
 ******************************************************************************/

extern usb_status_t USB_DeviceSetSpeed(uint8_t speed);

#endif /* __USB_DEVICE_DESCRIPTOR_H__ */

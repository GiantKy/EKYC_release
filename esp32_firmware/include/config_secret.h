#ifndef CONFIG_SECRET_H
#define CONFIG_SECRET_H

// ============================================================================
// WIFI CREDENTIALS (SENSITIVE - DO NOT COMMIT TO PUBLIC REPO)
// File này được ignore bởi Git nhằm bảo mật thông tin mạng cá nhân.
// ============================================================================

#ifndef WIFI_SSID
#define WIFI_SSID "ESP32_NCKH"
#endif

#ifndef WIFI_PASSWORD
#define WIFI_PASSWORD "giaky123"
#endif

// ============================================================================
// AI SERVER & DEVICE CONFIG DEFAULTS
// ============================================================================
#ifndef DEFAULT_AI_SERVER_IP
#define DEFAULT_AI_SERVER_IP "192.168.137.1"
#endif

#ifndef DEFAULT_AI_SERVER_PORT
#define DEFAULT_AI_SERVER_PORT 8000
#endif

#ifndef DEFAULT_NODEJS_SERVER_PORT
#define DEFAULT_NODEJS_SERVER_PORT 3000
#endif

#ifndef DEFAULT_DEVICE_ID
#define DEFAULT_DEVICE_ID "ESP32_S3_GATE_01"
#endif

#endif // CONFIG_SECRET_H

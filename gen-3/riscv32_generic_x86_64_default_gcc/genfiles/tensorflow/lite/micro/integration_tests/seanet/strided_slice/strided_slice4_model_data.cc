#include <cstdint>

#include "tensorflow/lite/micro/integration_tests/seanet/strided_slice/strided_slice4_model_data.h"

alignas(16) const unsigned char g_strided_slice4_model_data[] = {0x1c,0x0,0x0,0x0,0x54,0x46,0x4c,0x33,0x0,0x0,0x0,0x0,0x0,0x0,0xe,0x0,0x14,0x0,0x10,0x0,0xc,0x0,0x8,0x0,0x0,0x0,0x4,0x0,0xe,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0x88,0x0,0x0,0x0,0x34,0x4,0x0,0x0,0x3,0x0,0x0,0x0,0x5,0x0,0x0,0x0,0x74,0x0,0x0,0x0,0x54,0x0,0x0,0x0,0x2c,0x0,0x0,0x0,0xc,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x14,0xfd,0xff,0xff,0xc6,0xff,0xff,0xff,0x4,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0xe2,0xff,0xff,0xff,0x4,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0x0,0x0,0x6,0x0,0x8,0x0,0x4,0x0,0x6,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0xfe,0xff,0xff,0xff,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x74,0xfd,0xff,0xff,0x1,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0xf8,0xfc,0xff,0xff,0x10,0x0,0x0,0x0,0x68,0x0,0x0,0x0,0x6c,0x0,0x0,0x0,0x70,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x14,0x0,0x0,0x0,0x0,0x0,0xe,0x0,0x14,0x0,0x0,0x0,0x10,0x0,0xc,0x0,0xb,0x0,0x4,0x0,0xe,0x0,0x0,0x0,0x18,0x0,0x0,0x0,0x0,0x0,0x0,0x20,0x1c,0x0,0x0,0x0,0x20,0x0,0x0,0x0,0x8,0x0,0xc,0x0,0x8,0x0,0x4,0x0,0x8,0x0,0x0,0x0,0xf,0x0,0x0,0x0,0xd,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x2,0x0,0x0,0x0,0x3,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x5,0x0,0x0,0x0,0x6c,0x2,0x0,0x0,0xdc,0x1,0x0,0x0,0x50,0x1,0x0,0x0,0xc0,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x4a,0xfe,0xff,0xff,0x0,0x0,0x0,0x1,0x14,0x0,0x0,0x0,0x4c,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x0,0x0,0x0,0x7,0x8c,0x0,0x0,0x0,0xac,0xfd,0xff,0xff,0x10,0x0,0x0,0x0,0x1c,0x0,0x0,0x0,0x20,0x0,0x0,0x0,0x24,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x6c,0x42,0xc3,0x3b,0x1,0x0,0x0,0x0,0xe5,0x40,0x43,0x43,0x1,0x0,0x0,0x0,0x5b,0xba,0xb,0xc3,0x44,0x0,0x0,0x0,0x73,0x74,0x72,0x65,0x61,0x6d,0x61,0x62,0x6c,0x65,0x5f,0x6d,0x6f,0x64,0x65,0x6c,0x5f,0x31,0x30,0x2f,0x75,0x6e,0x65,0x74,0x5f,0x30,0x2f,0x65,0x6e,0x63,0x6f,0x64,0x65,0x72,0x5f,0x31,0x2f,0x63,0x6f,0x6e,0x76,0x32,0x64,0x5f,0x34,0x78,0x34,0x5f,0x61,0x2f,0x63,0x6f,0x6e,0x76,0x2f,0x73,0x74,0x72,0x69,0x64,0x65,0x64,0x5f,0x73,0x6c,0x69,0x63,0x65,0x0,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x2,0x0,0x0,0x0,0x52,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0x2,0xff,0xff,0xff,0x0,0x0,0x0,0x1,0x14,0x0,0x0,0x0,0x14,0x0,0x0,0x0,0x3,0x0,0x0,0x0,0x0,0x0,0x0,0x2,0x6c,0x0,0x0,0x0,0xec,0xfe,0xff,0xff,0x5c,0x0,0x0,0x0,0x73,0x74,0x72,0x65,0x61,0x6d,0x61,0x62,0x6c,0x65,0x5f,0x6d,0x6f,0x64,0x65,0x6c,0x5f,0x31,0x30,0x2f,0x75,0x6e,0x65,0x74,0x5f,0x30,0x2f,0x64,0x65,0x63,0x6f,0x64,0x65,0x72,0x5f,0x30,0x2f,0x63,0x6f,0x6e,0x76,0x32,0x64,0x74,0x72,0x61,0x6e,0x73,0x70,0x6f,0x73,0x65,0x5f,0x33,0x78,0x34,0x2f,0x63,0x72,0x6f,0x70,0x5f,0x66,0x72,0x65,0x71,0x5f,0x64,0x69,0x6d,0x2f,0x73,0x74,0x72,0x69,0x64,0x65,0x64,0x5f,0x73,0x6c,0x69,0x63,0x65,0x2f,0x73,0x74,0x61,0x63,0x6b,0x5f,0x32,0x0,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x8e,0xff,0xff,0xff,0x0,0x0,0x0,0x1,0x14,0x0,0x0,0x0,0x14,0x0,0x0,0x0,0x2,0x0,0x0,0x0,0x0,0x0,0x0,0x2,0x50,0x0,0x0,0x0,0x78,0xff,0xff,0xff,0x42,0x0,0x0,0x0,0x73,0x74,0x72,0x65,0x61,0x6d,0x61,0x62,0x6c,0x65,0x5f,0x6d,0x6f,0x64,0x65,0x6c,0x5f,0x31,0x30,0x2f,0x75,0x6e,0x65,0x74,0x5f,0x30,0x2f,0x64,0x65,0x63,0x6f,0x64,0x65,0x72,0x5f,0x31,0x2f,0x63,0x6f,0x6e,0x76,0x32,0x64,0x5f,0x33,0x78,0x33,0x2f,0x63,0x6f,0x6e,0x76,0x2f,0x73,0x74,0x72,0x69,0x64,0x65,0x64,0x5f,0x73,0x6c,0x69,0x63,0x65,0x0,0x0,0x1,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x0,0x0,0x16,0x0,0x1c,0x0,0x18,0x0,0x17,0x0,0x10,0x0,0xc,0x0,0x8,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x7,0x0,0x16,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x18,0x0,0x0,0x0,0x18,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x2,0x54,0x0,0x0,0x0,0x4,0x0,0x4,0x0,0x4,0x0,0x0,0x0,0x42,0x0,0x0,0x0,0x73,0x74,0x72,0x65,0x61,0x6d,0x61,0x62,0x6c,0x65,0x5f,0x6d,0x6f,0x64,0x65,0x6c,0x5f,0x31,0x30,0x2f,0x75,0x6e,0x65,0x74,0x5f,0x30,0x2f,0x64,0x65,0x63,0x6f,0x64,0x65,0x72,0x5f,0x30,0x2f,0x63,0x6f,0x6e,0x76,0x32,0x64,0x5f,0x33,0x78,0x33,0x2f,0x63,0x6f,0x6e,0x76,0x2f,0x73,0x74,0x72,0x69,0x64,0x65,0x64,0x5f,0x73,0x6c,0x69,0x63,0x65,0x0,0x0,0x1,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x0,0x0,0x16,0x0,0x18,0x0,0x14,0x0,0x13,0x0,0x0,0x0,0xc,0x0,0x8,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x7,0x0,0x16,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x1c,0x0,0x0,0x0,0x54,0x0,0x0,0x0,0x0,0x0,0x0,0x7,0x90,0x0,0x0,0x0,0xc,0x0,0x14,0x0,0x10,0x0,0xc,0x0,0x8,0x0,0x4,0x0,0xc,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0x1c,0x0,0x0,0x0,0x20,0x0,0x0,0x0,0x24,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x6c,0x42,0xc3,0x3b,0x1,0x0,0x0,0x0,0xe5,0x40,0x43,0x43,0x1,0x0,0x0,0x0,0x5b,0xba,0xb,0xc3,0x3d,0x0,0x0,0x0,0x73,0x74,0x72,0x65,0x61,0x6d,0x61,0x62,0x6c,0x65,0x5f,0x6d,0x6f,0x64,0x65,0x6c,0x5f,0x31,0x30,0x2f,0x75,0x6e,0x65,0x74,0x5f,0x30,0x2f,0x65,0x6e,0x63,0x6f,0x64,0x65,0x72,0x5f,0x31,0x2f,0x63,0x6f,0x6e,0x76,0x32,0x64,0x5f,0x34,0x78,0x34,0x5f,0x61,0x2f,0x63,0x6f,0x6e,0x76,0x2f,0x63,0x6f,0x6e,0x63,0x61,0x74,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x4,0x0,0x0,0x0,0x52,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0x1,0x0,0x0,0x0,0x10,0x0,0x0,0x0,0xc,0x0,0x10,0x0,0xf,0x0,0x0,0x0,0x8,0x0,0x4,0x0,0xc,0x0,0x0,0x0,0x2d,0x0,0x0,0x0,0x2,0x0,0x0,0x0,0x0,0x0,0x0,0x2d};

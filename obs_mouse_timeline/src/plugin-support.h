#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <stdarg.h>

extern const char *PLUGIN_NAME;
extern const char *PLUGIN_VERSION;

void plugin_log(int log_level, const char *format, ...);
extern void blogva(int log_level, const char *format, va_list args);

#ifdef __cplusplus
}
#endif

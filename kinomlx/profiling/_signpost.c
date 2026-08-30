// Minimal os_signpost wrapper for ctypes use.
//
// os_signpost's emit API is macro based and needs the calling image's
// __dso_handle. Python and PyObjC cannot supply that contract directly, so
// KinoMLX keeps this deliberately small C boundary rather than depending on
// the stock Python logging surface, which can only read signpost records.

#include <os/log.h>
#include <os/signpost.h>
#include <stdint.h>

#define KINO_PROFILE_SUBSYSTEM "org.dmlsrc.kinomlx"

static os_log_t _kino_poi_log = NULL;

static inline void _kino_init(void) {
    if (!_kino_poi_log) {
        _kino_poi_log = os_log_create(KINO_PROFILE_SUBSYSTEM,
                                      OS_LOG_CATEGORY_POINTS_OF_INTEREST);
    }
}

uint64_t kino_signpost_id_generate(void) {
    _kino_init();
    return os_signpost_id_generate(_kino_poi_log);
}

int kino_signpost_enabled(void) {
    _kino_init();
    return os_signpost_enabled(_kino_poi_log) ? 1 : 0;
}

void kino_signpost_interval_begin(uint64_t sid, const char *message) {
    _kino_init();
    os_signpost_interval_begin(_kino_poi_log, sid, "phase", "%{public}s", message);
}

void kino_signpost_interval_end(uint64_t sid, const char *message) {
    _kino_init();
    os_signpost_interval_end(_kino_poi_log, sid, "phase", "%{public}s", message);
}

void kino_signpost_event(uint64_t sid, const char *message) {
    _kino_init();
    os_signpost_event_emit(_kino_poi_log, sid, "advance", "%{public}s", message);
}

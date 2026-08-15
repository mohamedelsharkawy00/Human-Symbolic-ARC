#pragma once
#include <cstdint>
#include <cstdlib>  // getenv_s / strtoul for ARC_WS_MAX_GB / ARC_COMMIT_GB
#include <filesystem>
#include <windows.h>
#undef TRANSPARENT  // conflicts with environment.hpp sentinel
// OS specific window and power management helpers

namespace arc {

inline void enter_fullscreen(HWND hwnd, int w, int h) {
  DEVMODE dm_screen_settings;
  memset(&dm_screen_settings, 0, sizeof(dm_screen_settings));
  dm_screen_settings.dmSize = sizeof(dm_screen_settings);
  EnumDisplaySettings(NULL, ENUM_CURRENT_SETTINGS, &dm_screen_settings);

  DEVMODE target_mode;
  memset(&target_mode, 0, sizeof(target_mode));
  target_mode.dmSize = sizeof(target_mode);

  bool found_mode = false;
  for (int i = 0; EnumDisplaySettings(NULL, i, &target_mode); ++i) {
    if (target_mode.dmPelsWidth == (uint32_t)w &&
        target_mode.dmPelsHeight == (uint32_t)h &&
        target_mode.dmBitsPerPel == 32) {
      if (target_mode.dmDisplayFrequency !=
          dm_screen_settings.dmDisplayFrequency) {
        dm_screen_settings = target_mode;
        found_mode = true;
        break;
      }
    }
  }

  dm_screen_settings.dmPelsWidth = w;
  dm_screen_settings.dmPelsHeight = h;
  dm_screen_settings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT |
                                (found_mode ? DM_DISPLAYFREQUENCY : 0);
  // Uncomment to enable exclusive fullscreen mode
   //ChangeDisplaySettings(&dm_screen_settings, CDS_FULLSCREEN);
  // No TOPMOST to allow Alt-Tab
  SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, w, h, SWP_SHOWWINDOW);
}

inline void disable_fullscreen_optimizations() {
  WCHAR exe_path[MAX_PATH];
  GetModuleFileNameW(NULL, exe_path, MAX_PATH);

  HKEY h_key;
  LPCWSTR key_path = L"Software\\Microsoft\\Windows "
                     L"NT\\CurrentVersion\\AppCompatFlags\\Layers";

  if (RegOpenKeyExW(HKEY_CURRENT_USER, key_path, 0, KEY_READ | KEY_SET_VALUE,
                    &h_key) == ERROR_SUCCESS) {
    LPCWSTR value_data = L"~ DISABLEDXMAXIMIZEDWINDOWEDMODE";
    WCHAR existing_data[256] = {0};
    DWORD data_size = sizeof(existing_data);
    if (RegQueryValueExW(h_key, exe_path, NULL, NULL, (LPBYTE)existing_data,
                         &data_size) != ERROR_SUCCESS ||
        lstrcmpW(existing_data, value_data) != 0) {
      RegSetValueExW(h_key, exe_path, 0, REG_SZ, (const BYTE *)value_data,
                     (lstrlenW(value_data) + 1) * sizeof(WCHAR));
    }
    RegCloseKey(h_key);
  } else if (RegCreateKeyExW(HKEY_CURRENT_USER, key_path, 0, NULL,
                             REG_OPTION_NON_VOLATILE, KEY_SET_VALUE, NULL,
                             &h_key, NULL) == ERROR_SUCCESS) {
    LPCWSTR value_data = L"~ DISABLEDXMAXIMIZEDWINDOWEDMODE";
    RegSetValueExW(h_key, exe_path, 0, REG_SZ, (const BYTE *)value_data,
                   (lstrlenW(value_data) + 1) * sizeof(WCHAR));
    RegCloseKey(h_key);
  }

  if (RegOpenKeyExW(HKEY_CURRENT_USER, L"System\\GameConfigStore", 0,
                    KEY_SET_VALUE, &h_key) == ERROR_SUCCESS) {
    DWORD behavior = 2; // Disable FSO
    RegSetValueExW(h_key, L"GameDVR_FSEBehaviorMode", 0, REG_DWORD,
                   (const BYTE *)&behavior, sizeof(DWORD));
    RegCloseKey(h_key);
  }
}

inline void disable_power_throttling() {
  PROCESS_POWER_THROTTLING_STATE power_throttling{};
  power_throttling.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  power_throttling.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  power_throttling.StateMask = 0; // Disable throttling

  SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling,
                        &power_throttling, sizeof(power_throttling));
  SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
}

// Two independent guards, serving two different goals:
//
//  1. WORKING-SET MAX (the real control): caps the PHYSICAL RAM ARC holds
//     resident. On breach Windows trims ARC's own pages — ARC may slow, but
//     the machine never pages other processes out, so the system stays
//     responsive. This is the "never slow my machine" guarantee. Default
//     8 GB (measured resident ~2.4 GB, analysis worst case ~3 GB, so 8 GB
//     is comfortable and still leaves 15+ GB physical for everything else).
//
//  2. COMMIT CAP (runaway guillotine, NOT a working limit): caps total
//     address commit. On breach the allocation FAILS and the process is
//     killed instead of filling the pagefile forever. Set HIGH (20 GB) so
//     it never false-trips on CUDA's large committed-but-never-resident
//     reservations (~5 GB) — only a genuine leak/runaway hits it.
//
// Both overridable without a rebuild:
//   ARC_WS_MAX_GB   working-set maximum  (0 = no working-set limit)
//   ARC_COMMIT_GB   commit cap           (0 = no commit cap)
inline SIZE_T env_gb(const char* name, SIZE_T default_gb) {
	char buf[32];
	size_t len = 0;
	if (getenv_s(&len, buf, sizeof(buf), name) == 0 && len > 1) {
		char* end = nullptr;
		unsigned long v = strtoul(buf, &end, 10);
		if (end != buf) return (SIZE_T)v;
	}
	return default_gb;
}

inline void enforce_memory_limit() {
	const SIZE_T GB = 1024ULL * 1024 * 1024;
	SIZE_T ws_max_gb  = env_gb("ARC_WS_MAX_GB", 12);
	SIZE_T commit_gb  = env_gb("ARC_COMMIT_GB", 15);

	HANDLE job = CreateJobObjectW(nullptr, nullptr);
	if (!job) return;

	JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
	DWORD flags = 0;

	if (commit_gb > 0) {
		flags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY;
		info.ProcessMemoryLimit = commit_gb * GB;
	}
	if (ws_max_gb > 0) {
		flags |= JOB_OBJECT_LIMIT_WORKINGSET;
		// Min must be > 0 and <= max; 64 MB floor is plenty for the OS to
		// keep ARC's hot pages resident.
		info.BasicLimitInformation.MinimumWorkingSetSize = 64ULL * 1024 * 1024;
		info.BasicLimitInformation.MaximumWorkingSetSize = ws_max_gb * GB;
	}

	info.BasicLimitInformation.LimitFlags = flags;
	SetInformationJobObject(job, JobObjectExtendedLimitInformation, &info, sizeof(info));
	AssignProcessToJobObject(job, GetCurrentProcess());
}

}

/*
 * OBS Mouse Timeline
 * Copyright (C) 2026 Recoil Reconstruction
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 */

#include <obs-frontend-api.h>
#include <obs-module.h>
#include <util/bmem.h>
#include <util/platform.h>

#include <Windows.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "plugin-support.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE(PLUGIN_NAME, "en-US")

namespace {

constexpr UINT WM_STOP_RAW_INPUT = WM_APP + 71;
constexpr uint64_t NO_FRAME = std::numeric_limits<uint64_t>::max();

struct MouseEvent {
	uint64_t event_index = 0;
	uint64_t os_time_ns = 0;
	uint64_t session_time_ns = 0;
	uint64_t obs_video_time_ns = 0;
	uint64_t video_frame_index = NO_FRAME;
	uint32_t obs_global_frame_count = 0;
	int output_frame_count = 0;
	int32_t dx_counts = 0;
	int32_t dy_counts = 0;
	double dt_us = 0.0;
	double vx_counts_s = 0.0;
	double vy_counts_s = 0.0;
	double speed_counts_s = 0.0;
	uint16_t movement_flags = 0;
	uint16_t button_flags = 0;
	int16_t wheel_delta = 0;
	bool left_button_down = false;
	uintptr_t device_handle = 0;
};

struct FrameEvent {
	uint64_t frame_index = 0;
	uint64_t os_time_ns = 0;
	uint64_t session_time_ns = 0;
	uint64_t obs_video_time_ns = 0;
	uint32_t obs_global_frame_count = 0;
	int output_frame_count = 0;
};

struct MarkerEvent {
	std::string name;
	uint64_t os_time_ns = 0;
	uint64_t obs_video_time_ns = 0;
	uint64_t frame_index = NO_FRAME;
	int output_frame_count = 0;
};

std::string csv_escape(const std::string &value)
{
	if (value.find_first_of(",\"\r\n") == std::string::npos)
		return value;
	std::string escaped = "\"";
	for (const char character : value) {
		if (character == '\"')
			escaped += "\"\"";
		else
			escaped += character;
	}
	escaped += '\"';
	return escaped;
}

std::string json_escape(const std::string &value)
{
	std::ostringstream output;
	for (const unsigned char character : value) {
		switch (character) {
		case '\\': output << "\\\\"; break;
		case '\"': output << "\\\""; break;
		case '\b': output << "\\b"; break;
		case '\f': output << "\\f"; break;
		case '\n': output << "\\n"; break;
		case '\r': output << "\\r"; break;
		case '\t': output << "\\t"; break;
		default:
			if (character < 0x20) {
				output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
				       << static_cast<int>(character) << std::dec;
			} else {
				output << character;
			}
		}
	}
	return output.str();
}

std::string button_names(uint16_t flags)
{
	std::vector<std::string> names;
	auto add = [&](uint16_t flag, const char *name) {
		if ((flags & flag) != 0)
			names.emplace_back(name);
	};
	add(RI_MOUSE_LEFT_BUTTON_DOWN, "left_down");
	add(RI_MOUSE_LEFT_BUTTON_UP, "left_up");
	add(RI_MOUSE_RIGHT_BUTTON_DOWN, "right_down");
	add(RI_MOUSE_RIGHT_BUTTON_UP, "right_up");
	add(RI_MOUSE_MIDDLE_BUTTON_DOWN, "middle_down");
	add(RI_MOUSE_MIDDLE_BUTTON_UP, "middle_up");
	add(RI_MOUSE_BUTTON_4_DOWN, "button4_down");
	add(RI_MOUSE_BUTTON_4_UP, "button4_up");
	add(RI_MOUSE_BUTTON_5_DOWN, "button5_down");
	add(RI_MOUSE_BUTTON_5_UP, "button5_up");
	add(RI_MOUSE_WHEEL, "wheel");
	add(RI_MOUSE_HWHEEL, "hwheel");
	std::ostringstream output;
	for (size_t index = 0; index < names.size(); ++index) {
		if (index != 0)
			output << '|';
		output << names[index];
	}
	return output.str();
}

class MouseTimelineRecorder {
public:
	bool initialize()
	{
		raw_thread_ = std::thread(&MouseTimelineRecorder::raw_input_thread, this);
		{
			std::unique_lock<std::mutex> lock(raw_ready_mutex_);
			if (!raw_ready_cv_.wait_for(lock, std::chrono::seconds(3), [&] { return raw_ready_; })) {
				plugin_log(LOG_ERROR, "Raw Input window did not initialize within three seconds");
				return false;
			}
		}
		if (!raw_ready_success_) {
			plugin_log(LOG_ERROR, "Raw Input registration failed with Win32 error %lu",
				   static_cast<unsigned long>(raw_error_));
			return false;
		}

		obs_add_tick_callback(&MouseTimelineRecorder::tick_callback, this);
		obs_frontend_add_event_callback(&MouseTimelineRecorder::frontend_callback, this);
		plugin_log(LOG_INFO, "Ready: passive Raw Input and OBS video-tick capture enabled");
		return true;
	}

	void shutdown()
	{
		obs_frontend_remove_event_callback(&MouseTimelineRecorder::frontend_callback, this);
		obs_remove_tick_callback(&MouseTimelineRecorder::tick_callback, this);
		disconnect_output();
		active_.store(false, std::memory_order_release);
		if (raw_hwnd_)
			PostMessageW(raw_hwnd_, WM_STOP_RAW_INPUT, 0, 0);
		if (raw_thread_.joinable())
			raw_thread_.join();
	}

private:
	static void tick_callback(void *data, float)
	{
		static_cast<MouseTimelineRecorder *>(data)->on_tick();
	}

	static void frontend_callback(enum obs_frontend_event event, void *data)
	{
		static_cast<MouseTimelineRecorder *>(data)->on_frontend_event(event);
	}

	static void output_activate(void *data, calldata_t *)
	{
		static_cast<MouseTimelineRecorder *>(data)->begin_session("output_activate");
	}

	static void output_deactivate(void *data, calldata_t *)
	{
		static_cast<MouseTimelineRecorder *>(data)->end_session("output_deactivate");
	}

	void on_frontend_event(enum obs_frontend_event event)
	{
		switch (event) {
		case OBS_FRONTEND_EVENT_RECORDING_STARTING:
			connect_recording_output();
			add_marker("frontend_recording_starting");
			break;
		case OBS_FRONTEND_EVENT_RECORDING_STARTED:
			if (!active_.load(std::memory_order_acquire))
				begin_session("frontend_started_fallback");
			add_marker("frontend_recording_started");
			break;
		case OBS_FRONTEND_EVENT_RECORDING_PAUSED:
			add_marker("frontend_recording_paused");
			break;
		case OBS_FRONTEND_EVENT_RECORDING_UNPAUSED:
			add_marker("frontend_recording_unpaused");
			break;
		case OBS_FRONTEND_EVENT_RECORDING_STOPPING:
			add_marker("frontend_recording_stopping");
			break;
		case OBS_FRONTEND_EVENT_RECORDING_STOPPED:
			if (active_.load(std::memory_order_acquire))
				end_session("frontend_stopped_fallback");
			add_marker("frontend_recording_stopped");
			write_sidecars();
			disconnect_output();
			break;
		default: break;
		}
	}

	void connect_recording_output()
	{
		disconnect_output();
		recording_output_ = obs_frontend_get_recording_output();
		if (!recording_output_) {
			plugin_log(LOG_WARNING, "Recording output was unavailable at STARTING");
			return;
		}
		signal_handler_t *signals = obs_output_get_signal_handler(recording_output_);
		signal_handler_connect(signals, "activate", &MouseTimelineRecorder::output_activate, this);
		signal_handler_connect(signals, "deactivate", &MouseTimelineRecorder::output_deactivate, this);
		output_signals_connected_ = true;
	}

	void disconnect_output()
	{
		if (!recording_output_)
			return;
		if (output_signals_connected_) {
			signal_handler_t *signals = obs_output_get_signal_handler(recording_output_);
			signal_handler_disconnect(signals, "activate", &MouseTimelineRecorder::output_activate,
					  this);
			signal_handler_disconnect(signals, "deactivate",
					  &MouseTimelineRecorder::output_deactivate, this);
			output_signals_connected_ = false;
		}
		obs_output_release(recording_output_);
		recording_output_ = nullptr;
	}

	void begin_session(const char *origin)
	{
		bool expected = false;
		if (!active_.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
			return;

		const uint64_t now = os_gettime_ns();
		{
			std::lock_guard<std::mutex> lock(data_mutex_);
			mouse_events_.clear();
			frame_events_.clear();
			markers_.clear();
			session_start_ns_ = now;
			session_stop_ns_ = 0;
			activation_origin_ = origin;
			start_obs_global_frames_ = obs_get_total_frames();
			start_obs_video_time_ns_ = obs_get_video_frame_time();
			output_frames_at_start_ = current_output_frames();
			output_frames_at_stop_ = 0;
			output_dropped_at_stop_ = 0;
			left_button_down_ = false;
			last_mouse_ns_ = 0;
		}
		last_frame_index_.store(NO_FRAME, std::memory_order_release);
		last_video_time_ns_.store(start_obs_video_time_ns_, std::memory_order_release);
		last_obs_global_frames_.store(start_obs_global_frames_, std::memory_order_release);
		last_output_frames_.store(output_frames_at_start_, std::memory_order_release);
		add_marker("session_begin");
		plugin_log(LOG_INFO, "Mouse timeline session started via %s", origin);
	}

	void end_session(const char *origin)
	{
		if (!active_.exchange(false, std::memory_order_acq_rel))
			return;
		std::lock_guard<std::mutex> lock(data_mutex_);
		session_stop_ns_ = os_gettime_ns();
		output_frames_at_stop_ = current_output_frames();
		output_dropped_at_stop_ = recording_output_ ? obs_output_get_frames_dropped(recording_output_) : 0;
		stop_origin_ = origin;
		plugin_log(LOG_INFO, "Mouse timeline session stopped via %s", origin);
	}

	int current_output_frames() const
	{
		return recording_output_ ? obs_output_get_total_frames(recording_output_) : 0;
	}

	void on_tick()
	{
		if (!active_.load(std::memory_order_acquire))
			return;
		FrameEvent event;
		event.os_time_ns = os_gettime_ns();
		event.obs_video_time_ns = obs_get_video_frame_time();
		event.obs_global_frame_count = obs_get_total_frames();
		event.output_frame_count = current_output_frames();
		{
			std::lock_guard<std::mutex> lock(data_mutex_);
			if (!active_.load(std::memory_order_relaxed))
				return;
			event.frame_index = frame_events_.size();
			event.session_time_ns = event.os_time_ns - session_start_ns_;
			frame_events_.push_back(event);
		}
		last_frame_index_.store(event.frame_index, std::memory_order_release);
		last_video_time_ns_.store(event.obs_video_time_ns, std::memory_order_release);
		last_obs_global_frames_.store(event.obs_global_frame_count, std::memory_order_release);
		last_output_frames_.store(event.output_frame_count, std::memory_order_release);
	}

	void add_marker(const char *name)
	{
		MarkerEvent marker;
		marker.name = name;
		marker.os_time_ns = os_gettime_ns();
		marker.obs_video_time_ns = last_video_time_ns_.load(std::memory_order_acquire);
		marker.frame_index = last_frame_index_.load(std::memory_order_acquire);
		marker.output_frame_count = current_output_frames();
		std::lock_guard<std::mutex> lock(data_mutex_);
		markers_.push_back(std::move(marker));
	}

	void record_mouse(const RAWINPUT &input)
	{
		if (!active_.load(std::memory_order_acquire) || input.header.dwType != RIM_TYPEMOUSE)
			return;

		const RAWMOUSE &mouse = input.data.mouse;
		MouseEvent event;
		event.os_time_ns = os_gettime_ns();
		event.obs_video_time_ns = last_video_time_ns_.load(std::memory_order_acquire);
		event.video_frame_index = last_frame_index_.load(std::memory_order_acquire);
		event.obs_global_frame_count = last_obs_global_frames_.load(std::memory_order_acquire);
		event.output_frame_count = last_output_frames_.load(std::memory_order_acquire);
		event.dx_counts = mouse.lLastX;
		event.dy_counts = mouse.lLastY;
		event.movement_flags = mouse.usFlags;
		event.button_flags = mouse.usButtonFlags;
		event.wheel_delta = (mouse.usButtonFlags & (RI_MOUSE_WHEEL | RI_MOUSE_HWHEEL))
					    ? static_cast<int16_t>(mouse.usButtonData)
					    : 0;
		event.device_handle = reinterpret_cast<uintptr_t>(input.header.hDevice);

		std::lock_guard<std::mutex> lock(data_mutex_);
		if (!active_.load(std::memory_order_relaxed))
			return;
		if ((event.button_flags & RI_MOUSE_LEFT_BUTTON_DOWN) != 0)
			left_button_down_ = true;
		if ((event.button_flags & RI_MOUSE_LEFT_BUTTON_UP) != 0)
			left_button_down_ = false;
		event.left_button_down = left_button_down_;
		event.event_index = mouse_events_.size();
		event.session_time_ns = event.os_time_ns - session_start_ns_;
		if (last_mouse_ns_ != 0 && event.os_time_ns > last_mouse_ns_) {
			const double dt_s = static_cast<double>(event.os_time_ns - last_mouse_ns_) / 1e9;
			event.dt_us = dt_s * 1e6;
			event.vx_counts_s = static_cast<double>(event.dx_counts) / dt_s;
			event.vy_counts_s = static_cast<double>(event.dy_counts) / dt_s;
			event.speed_counts_s = std::hypot(event.vx_counts_s, event.vy_counts_s);
		}
		last_mouse_ns_ = event.os_time_ns;
		mouse_events_.push_back(event);
	}

	std::filesystem::path sidecar_base_path() const
	{
		const char *last_recording = obs_frontend_get_last_recording();
		if (last_recording && *last_recording) {
			std::filesystem::path path = std::filesystem::u8path(last_recording);
			return path.parent_path() / path.stem();
		}
		char *fallback = obs_module_config_path("orphan-recording");
		std::filesystem::path path = fallback ? std::filesystem::u8path(fallback)
						      : std::filesystem::path("orphan-recording");
		bfree(fallback);
		return path;
	}

	void write_sidecars()
	{
		std::vector<MouseEvent> mouse_events;
		std::vector<FrameEvent> frame_events;
		std::vector<MarkerEvent> markers;
		uint64_t start_ns;
		uint64_t stop_ns;
		uint32_t start_global_frames;
		uint64_t start_video_time;
		int output_start;
		int output_stop;
		int output_dropped;
		std::string activation_origin;
		std::string stop_origin;
		{
			std::lock_guard<std::mutex> lock(data_mutex_);
			mouse_events = mouse_events_;
			frame_events = frame_events_;
			markers = markers_;
			start_ns = session_start_ns_;
			stop_ns = session_stop_ns_ ? session_stop_ns_ : os_gettime_ns();
			start_global_frames = start_obs_global_frames_;
			start_video_time = start_obs_video_time_ns_;
			output_start = output_frames_at_start_;
			output_stop = output_frames_at_stop_;
			output_dropped = output_dropped_at_stop_;
			activation_origin = activation_origin_;
			stop_origin = stop_origin_;
		}

		const std::filesystem::path base = sidecar_base_path();
		std::error_code error;
		std::filesystem::create_directories(base.parent_path(), error);
		const auto mouse_path = std::filesystem::path(base.string() + ".mouse.csv");
		const auto frames_path = std::filesystem::path(base.string() + ".frames.csv");
		const auto session_path = std::filesystem::path(base.string() + ".mouse-session.json");

		write_mouse_csv(mouse_path, mouse_events);
		write_frames_csv(frames_path, frame_events);
		write_session_json(session_path, mouse_path, frames_path, mouse_events, frame_events,
				   markers, start_ns, stop_ns, start_global_frames, start_video_time,
				   output_start, output_stop, output_dropped, activation_origin,
				   stop_origin);
		plugin_log(LOG_INFO, "Wrote %zu mouse packets and %zu OBS ticks beside recording: %s",
			   mouse_events.size(), frame_events.size(), session_path.u8string().c_str());
	}

	static void write_mouse_csv(const std::filesystem::path &path,
				    const std::vector<MouseEvent> &events)
	{
		std::ofstream output(path, std::ios::binary);
		output << "event_index,os_time_ns,session_time_ns,obs_video_time_ns,video_frame_index,"
			  "obs_global_frame_count,output_frame_count,dx_counts,dy_counts,dt_us,"
			  "vx_counts_s,vy_counts_s,speed_counts_s,movement_flags,button_flags,"
			  "button_names,wheel_delta,left_button_down,device_handle\n";
		output << std::setprecision(12);
		for (const auto &event : events) {
			output << event.event_index << ',' << event.os_time_ns << ',' << event.session_time_ns
			       << ',' << event.obs_video_time_ns << ',';
			if (event.video_frame_index != NO_FRAME)
				output << event.video_frame_index;
			output << ',' << event.obs_global_frame_count << ',' << event.output_frame_count << ','
			       << event.dx_counts << ',' << event.dy_counts << ',' << event.dt_us << ','
			       << event.vx_counts_s << ',' << event.vy_counts_s << ','
			       << event.speed_counts_s << ',' << event.movement_flags << ','
			       << event.button_flags << ',' << csv_escape(button_names(event.button_flags))
			       << ',' << event.wheel_delta << ','
			       << (event.left_button_down ? "true" : "false") << ",0x" << std::hex
			       << event.device_handle << std::dec << '\n';
		}
	}

	static void write_frames_csv(const std::filesystem::path &path,
				     const std::vector<FrameEvent> &frames)
	{
		std::ofstream output(path, std::ios::binary);
		output << "frame_index,os_time_ns,session_time_ns,obs_video_time_ns,"
			  "obs_global_frame_count,output_frame_count\n";
		for (const auto &frame : frames) {
			output << frame.frame_index << ',' << frame.os_time_ns << ',' << frame.session_time_ns
			       << ',' << frame.obs_video_time_ns << ',' << frame.obs_global_frame_count << ','
			       << frame.output_frame_count << '\n';
		}
	}

	static void write_session_json(const std::filesystem::path &path,
				       const std::filesystem::path &mouse_path,
				       const std::filesystem::path &frames_path,
				       const std::vector<MouseEvent> &mouse_events,
				       const std::vector<FrameEvent> &frame_events,
				       const std::vector<MarkerEvent> &markers, uint64_t start_ns,
				       uint64_t stop_ns, uint32_t start_global_frames,
				       uint64_t start_video_time, int output_start, int output_stop,
				       int output_dropped, const std::string &activation_origin,
				       const std::string &stop_origin)
	{
		obs_video_info video_info{};
		obs_get_video_info(&video_info);
		std::ofstream output(path, std::ios::binary);
		output << "{\n"
		       << "  \"schema_version\": 1,\n"
		       << "  \"plugin\": \"" << PLUGIN_NAME << "\",\n"
		       << "  \"plugin_version\": \"" << PLUGIN_VERSION << "\",\n"
		       << "  \"obs_version\": \"" << json_escape(obs_get_version_string()) << "\",\n"
		       << "  \"clock\": \"libobs os_gettime_ns / obs_get_video_frame_time\",\n"
		       << "  \"activation_origin\": \"" << json_escape(activation_origin) << "\",\n"
		       << "  \"stop_origin\": \"" << json_escape(stop_origin) << "\",\n"
		       << "  \"session_start_os_time_ns\": " << start_ns << ",\n"
		       << "  \"session_stop_os_time_ns\": " << stop_ns << ",\n"
		       << "  \"duration_s\": " << std::setprecision(12)
		       << static_cast<double>(stop_ns - start_ns) / 1e9 << ",\n"
		       << "  \"start_obs_video_time_ns\": " << start_video_time << ",\n"
		       << "  \"start_obs_global_frame_count\": " << start_global_frames << ",\n"
		       << "  \"fps_num\": " << video_info.fps_num << ",\n"
		       << "  \"fps_den\": " << video_info.fps_den << ",\n"
		       << "  \"output_width\": " << video_info.output_width << ",\n"
		       << "  \"output_height\": " << video_info.output_height << ",\n"
		       << "  \"output_frames_at_start\": " << output_start << ",\n"
		       << "  \"output_frames_at_stop\": " << output_stop << ",\n"
		       << "  \"output_dropped_frames\": " << output_dropped << ",\n"
		       << "  \"mouse_packet_count\": " << mouse_events.size() << ",\n"
		       << "  \"obs_tick_count\": " << frame_events.size() << ",\n"
		       << "  \"mouse_csv\": \"" << json_escape(mouse_path.u8string()) << "\",\n"
		       << "  \"frames_csv\": \"" << json_escape(frames_path.u8string()) << "\",\n"
		       << "  \"alignment\": \"Each mouse packet stores the most recent OBS tick frame_index and obs_video_time_ns. frames.csv maps that index to the recording output frame counter.\",\n"
		       << "  \"markers\": [\n";
		for (size_t index = 0; index < markers.size(); ++index) {
			const auto &marker = markers[index];
			output << "    {\"name\": \"" << json_escape(marker.name)
			       << "\", \"os_time_ns\": " << marker.os_time_ns
			       << ", \"obs_video_time_ns\": " << marker.obs_video_time_ns
			       << ", \"frame_index\": ";
			if (marker.frame_index == NO_FRAME)
				output << "null";
			else
				output << marker.frame_index;
			output << ", \"output_frame_count\": " << marker.output_frame_count << '}';
			if (index + 1 != markers.size())
				output << ',';
			output << '\n';
		}
		output << "  ]\n}\n";
	}

	static LRESULT CALLBACK raw_window_proc(HWND hwnd, UINT message, WPARAM wparam, LPARAM lparam)
	{
		auto *recorder = reinterpret_cast<MouseTimelineRecorder *>(
			GetWindowLongPtrW(hwnd, GWLP_USERDATA));
		if (message == WM_INPUT && recorder) {
			UINT size = 0;
			if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lparam), RID_INPUT, nullptr, &size,
					    sizeof(RAWINPUTHEADER)) == 0 &&
			    size >= sizeof(RAWINPUTHEADER)) {
				std::vector<uint8_t> bytes(size);
				if (GetRawInputData(reinterpret_cast<HRAWINPUT>(lparam), RID_INPUT,
						    bytes.data(), &size, sizeof(RAWINPUTHEADER)) == size) {
					const auto *input = reinterpret_cast<const RAWINPUT *>(bytes.data());
					recorder->record_mouse(*input);
				}
			}
		} else if (message == WM_STOP_RAW_INPUT) {
			DestroyWindow(hwnd);
			return 0;
		} else if (message == WM_DESTROY) {
			PostQuitMessage(0);
			return 0;
		}
		return DefWindowProcW(hwnd, message, wparam, lparam);
	}

	void raw_input_thread()
	{
		SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
		const HINSTANCE instance = GetModuleHandleW(nullptr);
		const wchar_t *class_name = L"ObsMouseTimelineRawInputWindow";
		WNDCLASSEXW window_class{};
		window_class.cbSize = sizeof(window_class);
		window_class.lpfnWndProc = &MouseTimelineRecorder::raw_window_proc;
		window_class.hInstance = instance;
		window_class.lpszClassName = class_name;
		if (!RegisterClassExW(&window_class) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS) {
			signal_raw_ready(false, GetLastError());
			return;
		}
		raw_hwnd_ = CreateWindowExW(0, class_name, L"OBS Mouse Timeline Raw Input", 0, 0, 0,
					    0, 0, nullptr, nullptr, instance, nullptr);
		if (!raw_hwnd_) {
			signal_raw_ready(false, GetLastError());
			return;
		}
		SetWindowLongPtrW(raw_hwnd_, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(this));
		RAWINPUTDEVICE device{};
		device.usUsagePage = 0x01;
		device.usUsage = 0x02;
		device.dwFlags = RIDEV_INPUTSINK;
		device.hwndTarget = raw_hwnd_;
		if (!RegisterRawInputDevices(&device, 1, sizeof(device))) {
			const DWORD error = GetLastError();
			DestroyWindow(raw_hwnd_);
			raw_hwnd_ = nullptr;
			signal_raw_ready(false, error);
			return;
		}
		signal_raw_ready(true, ERROR_SUCCESS);

		MSG message{};
		while (GetMessageW(&message, nullptr, 0, 0) > 0) {
			TranslateMessage(&message);
			DispatchMessageW(&message);
		}
		raw_hwnd_ = nullptr;
	}

	void signal_raw_ready(bool success, DWORD error)
	{
		std::lock_guard<std::mutex> lock(raw_ready_mutex_);
		raw_ready_success_ = success;
		raw_error_ = error;
		raw_ready_ = true;
		raw_ready_cv_.notify_all();
	}

	std::thread raw_thread_;
	HWND raw_hwnd_ = nullptr;
	std::mutex raw_ready_mutex_;
	std::condition_variable raw_ready_cv_;
	bool raw_ready_ = false;
	bool raw_ready_success_ = false;
	DWORD raw_error_ = ERROR_SUCCESS;

	std::atomic<bool> active_{false};
	std::atomic<uint64_t> last_frame_index_{NO_FRAME};
	std::atomic<uint64_t> last_video_time_ns_{0};
	std::atomic<uint32_t> last_obs_global_frames_{0};
	std::atomic<int> last_output_frames_{0};
	std::mutex data_mutex_;
	std::vector<MouseEvent> mouse_events_;
	std::vector<FrameEvent> frame_events_;
	std::vector<MarkerEvent> markers_;
	uint64_t session_start_ns_ = 0;
	uint64_t session_stop_ns_ = 0;
	uint64_t start_obs_video_time_ns_ = 0;
	uint64_t last_mouse_ns_ = 0;
	uint32_t start_obs_global_frames_ = 0;
	int output_frames_at_start_ = 0;
	int output_frames_at_stop_ = 0;
	int output_dropped_at_stop_ = 0;
	bool left_button_down_ = false;
	std::string activation_origin_;
	std::string stop_origin_;

	obs_output_t *recording_output_ = nullptr;
	bool output_signals_connected_ = false;
};

MouseTimelineRecorder *g_recorder = nullptr;

} // namespace

bool obs_module_load(void)
{
	g_recorder = new MouseTimelineRecorder();
	if (!g_recorder->initialize()) {
		g_recorder->shutdown();
		delete g_recorder;
		g_recorder = nullptr;
		return false;
	}
	plugin_log(LOG_INFO, "Loaded version %s", PLUGIN_VERSION);
	return true;
}

void obs_module_unload(void)
{
	if (g_recorder) {
		g_recorder->shutdown();
		delete g_recorder;
		g_recorder = nullptr;
	}
	plugin_log(LOG_INFO, "Unloaded");
}

const char *obs_module_description(void)
{
	return "Records Windows Raw Input mouse packets beside OBS recordings with an exact libobs video-tick timeline.";
}

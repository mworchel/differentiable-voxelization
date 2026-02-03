#pragma once

#include <cstdio>

namespace dvx
{

enum class LogLevel : unsigned int
{
    Trace = 0,
    Debug,
    Info,
    Warn,
    Error,
    Unknown
};

char const* get_log_level_string(LogLevel level);

void set_log_level(LogLevel level);

void set_log_output_stream(FILE* stream);

using PrintFunction = void (*)(char const*);

void set_log_output_function(PrintFunction func);

void log_message(LogLevel level, char const* format, ...);

// Returns a formatted string valid until the next call to format_message() in the same thread. Intended for immediate use only.
char const* format_message(char const* format, ...);

}
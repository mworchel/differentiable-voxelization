#include "log.hpp"

#include <cstdarg>
#include <cstdio>
#include <iostream>
#include <iterator>

namespace dvx
{

struct LogState
{
    LogLevel      minimum_level = LogLevel::Warn;
    FILE*         stream        = stdout;
    PrintFunction print         = nullptr;
};

LogState& get_log_state()
{
    static LogState state;
    return state;
}

char const* get_log_level_string(LogLevel level)
{
    static char const* strings[] = {"Trace",
                                    "Debug",
                                    "Info",
                                    "Warn",
                                    "Error"};

    static_assert(std::size(strings) == static_cast<unsigned int>(LogLevel::Unknown), "Number of `LogLevel` values does not match number of strings.");

    return strings[static_cast<unsigned int>(level)];
}

void set_log_level(LogLevel level)
{
    // TODO: Not thread-safe
    get_log_state().minimum_level = level;
}

void set_log_output_stream(FILE* stream)
{
    LogState& state = get_log_state();

    if (!stream)
    {
        // Return to the default state
        LogState default_state{};
        state.stream = default_state.stream;
        state.print  = default_state.print;
        return;
    }

    state.stream = stream;
    state.print  = nullptr;
}

void set_log_output_function(PrintFunction func)
{
    if (!func) // Return to the default state
        set_log_output_stream(nullptr);

    LogState& state = get_log_state();
    state.stream = nullptr;
    state.print  = func;
}

void log_message(LogLevel level, char const* format, ...)
{
    LogState const& state = get_log_state();

    // Do not emit the message if it's level is below the minimum
    if (static_cast<unsigned int>(level) < static_cast<unsigned int>(state.minimum_level))
        return;

    std::va_list args;
    va_start(args, format);
    if (state.stream)
    {
        std::fprintf(state.stream, "[%s] ", get_log_level_string(level));
        std::vfprintf(state.stream, format, args);
        std::putc('\n', state.stream);
    }
    else // use state.print
    {
        // TODO: Check for errors
        thread_local char buffer[1024] = {0};
        std::vsnprintf(buffer, std::size(buffer), format, args);
        state.print(format_message("[%s] %s", get_log_level_string(level), buffer));
    }
    va_end(args);
}

char const* format_message(char const* format, ...)
{
    thread_local char buffer[1024] = {0};

    std::va_list args;
    va_start(args, format);
    int count = vsnprintf(buffer, std::size(buffer), format, args);
    va_end(args);

    if (count < 0 || count >= static_cast<int>(std::size(buffer)))
    {
        // Handle encoding error/truncation
        buffer[std::size(buffer) - 1] = '\0';
    }

    return buffer;
}

} // namespace dvx
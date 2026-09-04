#pragma once
// One printf into a string, so no call site has to own a buffer.
// Owning the buffer is the only reason this file exists -- the formatting
// itself is libc's, which every compiler this project builds on already has.
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>

// Let the compiler check every format string against its arguments, which is
// the one thing printf cannot do for itself.
#if defined(_MSC_VER)
#include <sal.h>
#define ARC_FORMAT_ARG _Printf_format_string_
#define ARC_FORMAT_CHECK
#else
#define ARC_FORMAT_ARG
#define ARC_FORMAT_CHECK __attribute__((format(printf, 1, 2)))
#endif

namespace arc {
using namespace std;

// Measure first, then write, so no value is ever too big for the buffer.
inline string to_text(ARC_FORMAT_ARG const char *form, ...) ARC_FORMAT_CHECK;
inline string to_text(const char *form, ...) {
  va_list args;
  va_start(args, form);
  va_list measure;
  va_copy(measure, args);
  int length = vsnprintf(nullptr, 0, form, measure);
  va_end(measure);
  string text(length > 0 ? length : 0, '\0');
  if (length > 0) vsnprintf(text.data(), (size_t)length + 1, form, args);
  va_end(args);
  return text;
}

// The shortest text that still reads back as the same number. printf has no
// spelling for it, so ask for more digits until the number survives the trip
// home. A float is asked as a float, or it would answer with a double's tail.
// PLAIN DIGITS BEAT AN EXPONENT: %g reaches for one the moment a value outgrows
// the digits asked for, so a score of 2530 came back as 2.53e+03 -- the same
// number said worse, and longer. Keep looking for a plain spelling, and settle
// for an exponent only when no number of digits gives one.
inline string shortest(float value) {
  string with_exponent;
  for (int digits = 1; digits < 10; ++digits) {
    string text = to_text("%.*g", digits, (double)value);
    if (strtof(text.c_str(), nullptr) != value) continue;
    if (text.find('e') == string::npos) return text;
    if (with_exponent.empty()) with_exponent = text;
  }
  return with_exponent.empty() ? to_text("%.9g", (double)value) : with_exponent;
}

inline string shortest(double value) {
  string with_exponent;
  for (int digits = 1; digits < 18; ++digits) {
    string text = to_text("%.*g", digits, value);
    if (strtod(text.c_str(), nullptr) != value) continue;
    if (text.find('e') == string::npos) return text;
    if (with_exponent.empty()) with_exponent = text;
  }
  return with_exponent.empty() ? to_text("%.17g", value) : with_exponent;
}

// printf has every alignment the panel uses except centred, so that one is here.
// Odd leftovers go to the right, so a centred column never drifts left.
inline string align_centre(const string &text, int width) {
  int missing = width - (int)text.size();
  if (missing <= 0) return text;
  int before = missing / 2;
  return string(before, ' ') + text + string(missing - before, ' ');
}

}  // namespace arc

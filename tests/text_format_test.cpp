// Checks every format spelling the codebase actually uses. Silent when it
// passes; the build fails loudly when it does not.
#include "text_format.hpp"

#include <cstdio>
#include <string>

using namespace arc;
using namespace std;

static int failures = 0;

static void same(const string &got, const string &want, const char *label) {
  if (got == want) return;
  printf("FAIL %s: got \"%s\" want \"%s\"\n", label, got.c_str(), want.c_str());
  ++failures;
}

static void same_length(const string &got, size_t want, const char *label) {
  if (got.size() == want) return;
  printf("FAIL %s: got %zu chars, want %zu\n", label, got.size(), want);
  ++failures;
}

int main() {
  // Plain gluing, which is 166 of the codebase's 230 placeholders.
  same(to_text("%d", 42), "42", "int");
  same(to_text("%s", "abc"), "abc", "string");
  same(to_text("listening on 127.0.0.1:%d", 2026), "listening on 127.0.0.1:2026", "port line");

  // Decimal places.
  same(to_text("%.0f", 2.4), "2", "0 places");
  same(to_text("%.1f", 3.14159), "3.1", "1 place");
  same(to_text("%.2f", 3.14159), "3.14", "2 places");
  same(to_text("%.4f", 3.14159), "3.1416", "4 places");

  // Shortest text that reads back as the same number: no tail the value did not
  // earn, and no digit dropped that it needs.
  same(shortest(0.85f), "0.85", "shortest simple float");
  same(shortest(12.5f), "12.5", "shortest exact float");
  same(shortest(1.0f / 3.0f), "0.33333334", "shortest awkward float");
  same(shortest(1234567.0f), "1234567", "shortest big float stays plain");
  // Real scores off a solver log: an exponent here is the same number said worse.
  same(shortest(2530.0f), "2530", "round score stays plain");
  same(shortest(-40.0f), "-40", "round negative stays plain");
  same(shortest(2746.8f), "2746.8", "score keeps its tail");
  same(shortest(-44.0f), "-44", "small negative stays plain");
  // Too small for plain digits, so an exponent is the honest answer.
  same(shortest(1e-7f), "1e-07", "tiny float takes an exponent");
  same(shortest(0.1), "0.1", "shortest simple double");
  same(shortest(1.0 / 3.0), "0.3333333333333333", "shortest awkward double");
  if (strtof(shortest(1.0f / 3.0f).c_str(), nullptr) != 1.0f / 3.0f) {
    printf("FAIL float round trip\n");
    ++failures;
  }

  // Zero-padded integers.
  same(to_text("%02d", 7), "07", "2 digits");
  same(to_text("%03d", 7), "007", "3 digits");

  // Zero-padded decimals: the width counts the point and the places.
  same(to_text("%07.4f", 1.5), "01.5000", "7 wide 4 places");
  same(to_text("%07.2f", 1.5), "0001.50", "7 wide 2 places");
  same(to_text("%06.1f", 1.5), "0001.5", "6 wide 1 place");

  // Column alignment.
  same(to_text("%6s", "ab"), "    ab", "right align");
  same(to_text("%-3s", "a"), "a  ", "left align");
  same(to_text("%4s", "toolong"), "toolong", "too wide to align");

  // Width chosen at runtime, which fmt spelled {:0{}d}.
  same(to_text("%0*d", 4, 7), "0007", "runtime width");

  // Centring, the one alignment printf has no spelling for.
  same(align_centre("AB", 6), "  AB  ", "centre even");
  same(align_centre("ABC", 6), " ABC  ", "centre odd leans right");
  same(align_centre("TOOLONG", 3), "TOOLONG", "centre too wide");

  // A real converted log line, end to end.
  same(to_text("[BEAM %d] %.2f  \"%s\"", 3, 1.5, "x"), "[BEAM 3] 1.50  \"x\"", "beam line");

  // Nothing is ever too big for the buffer: no fixed size would have held these.
  same_length(to_text("%s", string(5000, 'x').c_str()), 5000, "long string");
  // Compared against what printf itself says it needs, since the thing under
  // test is the buffer, not printf's digits.
  same_length(to_text("%.2f", 1e300), (size_t)snprintf(nullptr, 0, "%.2f", 1e300), "huge double");

  if (failures > 0) printf("%d text format check(s) failed.\n", failures);
  return failures > 0 ? 1 : 0;
}

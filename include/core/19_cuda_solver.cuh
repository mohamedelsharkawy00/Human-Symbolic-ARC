#pragma once

bool cuda_init();

// How much of the card is already spoken for. Anything resident that is not us
// -- a game, a second solver, a browser -- shows up as missing free memory, and
// makes timings from this run incomparable to an uncontended one.
struct gpu_memory {
  unsigned long long free_mb = 0;
  unsigned long long total_mb = 0;
  unsigned long long used_mb = 0;
  int used_percent = 0;
};
gpu_memory read_gpu_memory();

#!/usr/bin/env bash
set -euo pipefail

arch="$(uname -m)"
case "$arch" in
  x86_64|amd64)
    simd_feature="avx"
    ;;
  arm64|aarch64)
    simd_feature="neon"
    ;;
  *)
    echo "Unsupported architecture: $arch" >&2
    echo "Expected x86_64/amd64 (avx) or arm64/aarch64 (neon)." >&2
    exit 1
    ;;
esac

bench_tiers=(fast good high extreme)
bench_files=()

for tier in "${bench_tiers[@]}"; do
  tier_files=(benches/ardftsrc_"$tier"_*.rs)
  for bench_file in "${tier_files[@]}"; do
    [[ -e "$bench_file" ]] || continue
    bench_files+=("$bench_file")
  done
done

if (( ${#bench_files[@]} == 0 )); then
  echo "No ardftsrc benchmark files found in benches/." >&2
  exit 1
fi

echo "Architecture: $arch"
echo "Using feature: $simd_feature"

for bench_file in "${bench_files[@]}"; do
  bench_target="$(basename "$bench_file" .rs)"
  echo
  echo "Running benchmark: $bench_target"
  /usr/bin/time -l cargo bench --features "$simd_feature" --bench "$bench_target" "$@"
done

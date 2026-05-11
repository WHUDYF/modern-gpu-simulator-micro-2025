#!/usr/bin/env bash
set -u

ROOT="${1:-/home/dyf/workloads/trace-compressions-industrial-codex-workload}"
SRC_DIR="$ROOT/sources"
STATUS="$ROOT/clone_status.tsv"
LOG_DIR="$ROOT/logs"

mkdir -p "$SRC_DIR" "$LOG_DIR"

printf "name\tstatus\tcommit\tpath\turl\n" > "$STATUS"

repos=(
  "gpu-rodinia|https://github.com/yuhc/gpu-rodinia.git"
  "gpu-parboil|https://github.com/yuhc/gpu-parboil.git"
  "shoc|https://github.com/vetter/shoc.git"
  "altis|https://github.com/utcs-scea/altis.git"
  "deepbench|https://github.com/baidu-research/DeepBench.git"
  "cutlass|https://github.com/NVIDIA/cutlass.git"
  "mlperf-inference|https://github.com/mlcommons/inference.git"
  "gunrock|https://github.com/gunrock/gunrock.git"
  "pannotia|https://github.com/pannotia/pannotia.git"
  "hecbench|https://github.com/zjin-lcf/HeCBench.git"
  "lammps|https://github.com/lammps/lammps.git"
  "gromacs|https://github.com/gromacs/gromacs.git"
)

for entry in "${repos[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  target="$SRC_DIR/$name"
  log="$LOG_DIR/${name}.log"

  if git -C "$target" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    commit="$(git -C "$target" rev-parse --short HEAD 2>/dev/null || printf unknown)"
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "exists" "$commit" "$target" "$url" >> "$STATUS"
    continue
  fi

  rm -rf "$target"
  printf "Cloning %s -> %s\n" "$name" "$target"
  if GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false timeout 30m git clone --depth 1 "$url" "$target" >"$log" 2>&1; then
    commit="$(git -C "$target" rev-parse --short HEAD 2>/dev/null || printf unknown)"
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "cloned" "$commit" "$target" "$url" >> "$STATUS"
  else
    code="$?"
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "failed:$code" "-" "$target" "$url" >> "$STATUS"
    printf "Failed %s; see %s\n" "$name" "$log"
  fi
done

printf "Wrote %s\n" "$STATUS"

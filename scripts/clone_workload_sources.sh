#!/usr/bin/env bash
set -u

ROOT="${1:-workloads/trace-compressions-industrial-codex-workload}"
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

declare -A sparse_roots=(
  ["mlperf-inference"]="language vision/classification_and_detection recommendation vision/medical_imaging text_to_image tools loadgen"
  ["hecbench"]="src"
)

apply_sparse_checkout() {
  local name="$1"
  local target="$2"
  local log="$3"
  if [[ -n "${sparse_roots[$name]:-}" ]]; then
    {
      git -C "$target" sparse-checkout init --cone
      git -C "$target" sparse-checkout set ${sparse_roots[$name]}
    } >>"$log" 2>&1
    return $?
  fi
  return 0
}

for entry in "${repos[@]}"; do
  name="${entry%%|*}"
  url="${entry#*|}"
  target="$SRC_DIR/$name"
  status_path="sources/$name"
  log="$LOG_DIR/${name}.log"

  if git -C "$target" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    if apply_sparse_checkout "$name" "$target" "$log"; then
      :
    else
      code="$?"
      printf "%s\t%s\t%s\t%s\t%s\n" "$name" "failed:sparse:$code" "-" "$status_path" "$url" >> "$STATUS"
      printf "Failed sparse checkout for existing %s; preserved %s; see %s\n" "$name" "$target" "$log"
      continue
    fi
    commit="$(git -C "$target" rev-parse --short HEAD 2>/dev/null || printf unknown)"
    status="exists"
    if [[ -n "${sparse_roots[$name]:-}" ]]; then
      status="sparse_partial"
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$status" "$commit" "$status_path" "$url" >> "$STATUS"
    continue
  fi

  rm -rf "$target"
  printf "Cloning %s -> %s\n" "$name" "$target"
  if [[ -n "${sparse_roots[$name]:-}" ]]; then
    clone_cmd=(git clone --depth 1 --filter=blob:none --sparse "$url" "$target")
  else
    clone_cmd=(git clone --depth 1 "$url" "$target")
  fi
  if GIT_TERMINAL_PROMPT=0 GIT_ASKPASS=/bin/false timeout 30m "${clone_cmd[@]}" >"$log" 2>&1; then
    if [[ -n "${sparse_roots[$name]:-}" ]]; then
      if apply_sparse_checkout "$name" "$target" "$log"; then
        :
      else
        code="$?"
        printf "%s\t%s\t%s\t%s\t%s\n" "$name" "failed:sparse:$code" "-" "$status_path" "$url" >> "$STATUS"
        printf "Failed sparse checkout for %s; see %s\n" "$name" "$log"
        rm -rf "$target"
        continue
      fi
    fi
    commit="$(git -C "$target" rev-parse --short HEAD 2>/dev/null || printf unknown)"
    status="cloned"
    if [[ -n "${sparse_roots[$name]:-}" ]]; then
      status="sparse_partial"
    fi
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$status" "$commit" "$status_path" "$url" >> "$STATUS"
  else
    code="$?"
    printf "%s\t%s\t%s\t%s\t%s\n" "$name" "failed:$code" "-" "$status_path" "$url" >> "$STATUS"
    printf "Failed %s; see %s\n" "$name" "$log"
    rm -rf "$target"
  fi
done

printf "Wrote %s\n" "$STATUS"

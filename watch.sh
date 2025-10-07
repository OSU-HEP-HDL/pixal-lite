#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden via -e ...)
WATCH_DIR="${WATCH_DIR:-/mount/machine_learning/validation}"
MODEL_DIR="${MODEL_DIR:-/mount/machine_learning/models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mount/machine_learning/results}"
PATTERN="${PATTERN:-jpg,jpeg,png,tif,tiff,bmp}"
CMD_TEMPLATE="${CMD_TEMPLATE:-pixal validate -i \"{component}\" -o \"{output}\"}"
SENTINEL_NAME="${SENTINEL_NAME:-.validated}"
RERUN_ON_CHANGE="${RERUN_ON_CHANGE:-false}"
READY_TIMEOUT="${READY_TIMEOUT:-60}"
READY_POLL_INTERVAL="${READY_POLL_INTERVAL:-0.5}"

DEFAULT_TEMPLATE='pixal validate -i "{component}" -o "{output}"'
export MODEL_DIR
# If you want to *force* the default and ignore any external env (recommended while debugging):
unset CMD_TEMPLATE
CMD_TEMPLATE="$DEFAULT_TEMPLATE"
echo "Command template: $CMD_TEMPLATE"

command -v inotifywait >/dev/null 2>&1 || { echo "inotifywait not found"; exit 1; }

echo "Watching: $WATCH_DIR"
echo "Extensions: $PATTERN"
echo "Rerun on change: $RERUN_ON_CHANGE"
echo "Command template: $CMD_TEMPLATE"

ext_regex="$(echo "$PATTERN" | tr ',' '|' )"
EVENTS="close_write,moved_to"

mkdir -p "$WATCH_DIR"

has_image_in_dir() {
  local d="$1"
  shopt -s nullglob nocaseglob
  local exts=(${PATTERN//,/ })
  for ext in "${exts[@]}"; do
    for f in "$d"/*."$ext"; do
      [[ -s "$f" ]] && return 0
    done
  done
  return 1
}

# serial_dir readiness: all immediate subdirs (or REQUIRED_DIRS) contain ≥1 image
component_ready() {
  local serial_dir="$1"

  if [[ -n "${REQUIRED_DIRS:-}" ]]; then
    IFS=',' read -ra reqs <<< "$REQUIRED_DIRS"
    for name in "${reqs[@]}"; do
      local d="$serial_dir/$name"
      [[ -d "$d" ]] || return 1
      has_image_in_dir "$d" || return 1
    done
    return 0
  fi

  local found_subdirs=0
  local d
  while IFS= read -r -d '' d; do
    case "$(basename "$d")" in
      .*|meta|logs|output|results) continue ;;
    esac
    found_subdirs=1
    has_image_in_dir "$d" || return 1
  done < <(find "$serial_dir" -mindepth 1 -maxdepth 1 -type d -print0)

  [[ "$found_subdirs" -ge 1 ]] || return 1
  return 0
}

# Poll for readiness with timeout
wait_until_ready() {
  local serial_dir="$1"
  local start now
  start=$(date +%s)
  while true; do
    if component_ready "$serial_dir"; then
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= READY_TIMEOUT )); then
      return 1
    fi
    sleep "$READY_POLL_INTERVAL"
  done
}

# Map a file to its <serial> directory: .../<component>/<serial>/<subdir>/<file>
get_serial_dir() {
  local file_path="$1"
  local subdir; subdir="$(dirname "$file_path")"   # .../<component>/<serial>/<subdir>
  dirname "$subdir"                                # .../<component>/<serial>
}

# Validate a single <serial> directory
run_validate_once() {
  local serial_dir="$1"
  local component_name; component_name="$(basename "$(dirname "$serial_dir")")"
  local serial_name;    serial_name="$(basename "$serial_dir")"

  local output="${OUTPUT_ROOT%/}/${component_name}/${serial_name}"
  mkdir -p "$output"

  local lockdir="$serial_dir/.validating.lock"
  local sentinel="$output/$SENTINEL_NAME"

  if [[ "$RERUN_ON_CHANGE" != "true" && -e "$sentinel" ]]; then
    echo "[Watcher] Skipping (already validated): $serial_dir"
    return 0
  fi

  if mkdir "$lockdir" 2>/dev/null; then
    trap 'rmdir "$lockdir" 2>/dev/null || true' EXIT

    # Default template we consider “safe”
    local default_tmpl='pixal validate -i "{component}" -o "{output}"'

    # If template is exactly the default (most cases), run without eval
    if [[ "${CMD_TEMPLATE}" == "${default_tmpl}" || -z "${CMD_TEMPLATE}" ]]; then
      echo "[Watcher] Validating:"
      echo "  Component: $component_name"
      echo "  Serial:    $serial_name"
      echo "  Output:    $output"
      echo "  Command:   pixal validate -i \"$serial_dir\" -o \"$output\""
      if pixal validate -i "$serial_dir" -o "$output"; then
        : > "$sentinel"
        echo "[Watcher] Done: $serial_dir"
      else
        echo "[Watcher] Command failed for: $serial_dir"
      fi
      rmdir "$lockdir" 2>/dev/null || true
      trap - EXIT
      return
    fi

    # For *custom* templates, do a quick sanity check then eval
    local cmd="${CMD_TEMPLATE//\{component\}/$serial_dir}"
    cmd="${cmd//\{output\}/$output}"

    # Sanity: require an even number of double quotes
    if (( $(printf '%s' "$cmd" | tr -cd '"' | wc -c) % 2 )); then
      echo "[Watcher] ERROR: Unbalanced quotes in CMD_TEMPLATE after substitution:"
      echo "  $cmd"
      rmdir "$lockdir" 2>/dev/null || true
      trap - EXIT
      return 1
    fi

    echo "[Watcher] Validating:"
    echo "  Component: $component_name"
    echo "  Serial:    $serial_name"
    echo "  Output:    $output"
    echo "  Command:   $cmd"

    if eval -- "$cmd"; then
      : > "$sentinel"
      echo "[Watcher] Done: $serial_dir"
    else
      echo "[Watcher] Command failed for: $serial_dir"
    fi

    rmdir "$lockdir" 2>/dev/null || true
    trap - EXIT
  else
    echo "[Watcher] Another validation in progress for: $serial_dir"
  fi
}

# Main loop: watch only the validation root
inotifywait -m --recursive --format '%e|%w|%f' -e "$EVENTS" "$WATCH_DIR" | \
while IFS='|' read -r event dir file; do
  if [[ "${file,,}" =~ \.(${ext_regex,,})$ ]]; then
    path="${dir}${file}"
    if [[ -s "$path" ]]; then
      serial_dir="$(get_serial_dir "$path")"
      sleep 0.1
      if wait_until_ready "$serial_dir"; then
        run_validate_once "$serial_dir"
      else
        echo "[Watcher] Timed out waiting for readiness: $serial_dir"
      fi
    fi
  fi
done

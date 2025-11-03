#!/usr/bin/env bash
set -euo pipefail

# --- Config (env-overridable) ---
WATCH_DIR="${WATCH_DIR:-/mount/machine_learning/validation}"
MODEL_DIR="${MODEL_DIR:-/mount/machine_learning/models}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mount/machine_learning/results}"
PATTERN="${PATTERN:-jpg,jpeg,png,tif,tiff,bmp}"
CMD_TEMPLATE="${CMD_TEMPLATE:-pixal validate -i \"{component}\" -o \"{output}\"}"
SENTINEL_NAME="${SENTINEL_NAME:-.validated}"
RERUN_ON_CHANGE="${RERUN_ON_CHANGE:-false}"
READY_TIMEOUT="${READY_TIMEOUT:-60}"
READY_POLL_INTERVAL="${READY_POLL_INTERVAL:-0.5}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"   # used in polling mode
FORCE_DEFAULT_CMD_TEMPLATE="${FORCE_DEFAULT_CMD_TEMPLATE:-false}"
DONE_FILE_NAME="${DONE_FILE_NAME:-done.txt}"     # <--- NEW

export MODEL_DIR
WATCH_DIR="${WATCH_DIR%/}"
OUTPUT_ROOT="${OUTPUT_ROOT%/}"

DEFAULT_TEMPLATE='pixal validate -i "{component}" -o "{output}"'
if [[ "$FORCE_DEFAULT_CMD_TEMPLATE" == "true" ]]; then
  CMD_TEMPLATE="$DEFAULT_TEMPLATE"
fi

echo "Watching: $WATCH_DIR"
echo "Extensions: $PATTERN"
echo "Rerun on change: $RERUN_ON_CHANGE"
echo "Command template: $CMD_TEMPLATE"
echo "Done trigger file: $DONE_FILE_NAME"  # <--- NEW

# Build helpers for extension matching
ext_regex_lower="$(echo "$PATTERN" | tr 'A-Z,' 'a-z|' )"
ext_regex_emacs="$(echo "$PATTERN" | sed 's/,/\\|/g')"   # for find -iregex (Emacs syntax)

# --- NEW: purge helper: safely wipe contents of WATCH_DIR when done.txt is created ---
purge_watch_dir() {
  local wd="$WATCH_DIR"
  local df="$wd/$DONE_FILE_NAME"

  # Safety checks
  [[ -d "$wd" ]] || { echo "[Watcher] Purge skipped: WATCH_DIR does not exist."; return 0; }
  # Require the done file to be present at top level before purging
  [[ -f "$df" ]] || { echo "[Watcher] Purge skipped: done file not present."; return 0; }

  # Extra safety: never allow purging '/' or empty path
  if [[ -z "$wd" || "$wd" == "/" ]]; then
    echo "[Watcher] Refusing to purge: WATCH_DIR resolves to '$wd'"
    return 1
  fi

  echo "[Watcher] 'done' trigger detected — purging contents of: $wd"

  # Remove everything inside WATCH_DIR, including hidden files/dirs, but not '.' or '..'
  shopt -s nullglob dotglob
  for entry in "$wd"/*; do
    # Skip the done file itself; remove it last to avoid re-triggering while purging
    if [[ "$(basename "$entry")" == "$DONE_FILE_NAME" ]]; then
      continue
    fi
    rm -rf --one-file-system -- "$entry"
  done
  shopt -u dotglob

  # Finally remove the done file
  rm -f -- "$df"

  echo "[Watcher] Watch directory emptied."
}

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

  local found_subdirs=0 d
  while IFS= read -r -d '' d; do
    case "$(basename "$d")" in .*|meta|logs|output|results) continue ;; esac
    found_subdirs=1
    has_image_in_dir "$d" || return 1
  done < <(find "$serial_dir" -mindepth 1 -maxdepth 1 -type d -print0)

  [[ "$found_subdirs" -ge 1 ]]
}

wait_until_ready() {
  local serial_dir="$1" start now
  start=$(date +%s)
  while true; do
    if component_ready "$serial_dir"; then return 0; fi
    now=$(date +%s)
    (( now - start >= READY_TIMEOUT )) && return 1
    sleep "$READY_POLL_INTERVAL"
  done
}

get_serial_dir() {
  local file_path="$1"
  local subdir; subdir="$(dirname "$file_path")"   # .../<component>/<serial>/<subdir>
  dirname "$subdir"                                # .../<component>/<serial>
}

run_validate_once() {
  local serial_dir="$1"
  local component_name; component_name="$(basename "$(dirname "$serial_dir")")"
  local serial_name;    serial_name="$(basename "$serial_dir")"

  local output="${OUTPUT_ROOT}/${component_name}/${serial_name}"
  mkdir -p "$output"

  local lockdir="$serial_dir/.validating.lock"
  local sentinel="$output/$SENTINEL_NAME"

  if [[ "$RERUN_ON_CHANGE" != "true" && -e "$sentinel" ]]; then
    echo "[Watcher] Skipping (already validated): $serial_dir"
    return 0
  fi

  if mkdir "$lockdir" 2>/dev/null; then
    trap 'rmdir "$lockdir" 2>/dev/null || true' EXIT

    if [[ "$CMD_TEMPLATE" == "$DEFAULT_TEMPLATE" || -z "$CMD_TEMPLATE" ]]; then
      echo "[Watcher] Validating:"
      echo "  Component: $component_name"
      echo "  Serial:    $serial_name"
      echo "  Output:    $output"
      echo "  Command:   pixal validate -i \"$serial_dir\" -o \"$output\""
      if pixal validate -i "$serial_dir" -o "$output"; then
        : > "$sentinel"; echo "[Watcher] Done: $serial_dir"
      else
        echo "[Watcher] Command failed for: $serial_dir"
      fi
    else
      local cmd="${CMD_TEMPLATE//\{component\}/$serial_dir}"
      cmd="${cmd//\{output\}/$output}"
      if (( $(printf '%s' "$cmd" | tr -cd '"' | wc -c) % 2 )); then
        echo "[Watcher] ERROR: Unbalanced quotes in CMD_TEMPLATE after substitution:"; echo "  $cmd"
      else
        echo "[Watcher] Validating:"
        echo "  Component: $component_name"
        echo "  Serial:    $serial_name"
        echo "  Output:    $output"
        echo "  Command:   $cmd"
        if eval -- "$cmd"; then
          : > "$sentinel"; echo "[Watcher] Done: $serial_dir"
        else
          echo "[Watcher] Command failed for: $serial_dir"
        fi
      fi
    fi

    rmdir "$lockdir" 2>/dev/null || true
    trap - EXIT
  else
    echo "[Watcher] Another validation in progress for: $serial_dir"
  fi
}

do_watch_inotify() {
  echo "[Watcher] Mode: inotify"
  command -v inotifywait >/dev/null 2>&1 || { echo "inotifywait not found"; return 1; }
  local EVENTS="close_write,moved_to"
  mkdir -p "$WATCH_DIR"
  inotifywait -m --recursive --format '%e|%w|%f' -e "$EVENTS" "$WATCH_DIR" | \
  while IFS='|' read -r event dir file; do
    # --- NEW: react to top-level done.txt creation ---
    if [[ "$dir" == "$WATCH_DIR/" && "$file" == "$DONE_FILE_NAME" ]]; then
      purge_watch_dir
      continue
    fi
    # -------------------------------------------------

    if [[ "${file,,}" =~ \.(${ext_regex_lower})$ ]]; then
      local path="${dir}${file}"
      if [[ -s "$path" ]]; then
        local serial_dir; serial_dir="$(get_serial_dir "$path")"
        sleep 0.1
        if wait_until_ready "$serial_dir"; then
          run_validate_once "$serial_dir"
        else
          echo "[Watcher] Timed out waiting for readiness: $serial_dir"
        fi
      fi
    fi
  done
}

do_watch_polling() {
  echo "[Watcher] Mode: polling (NFS/SMB or no inotify)"
  mkdir -p "$WATCH_DIR"
  local STATE="/tmp/pixal_poll_state.tsv"
  touch "$STATE"
  while :; do
    # --- NEW: polling check for done.txt trigger ---
    if [[ -f "$WATCH_DIR/$DONE_FILE_NAME" ]]; then
      purge_watch_dir
    fi
    # ----------------------------------------------

    # Find newest timestamp per serial_dir that has matching files
    declare -A newest=()
    while IFS= read -r line; do
      # line format: "<epoch> <dir_of_file>"
      local ts dir
      ts="${line%% *}"; dir="${line#* }"
      [[ -n "$dir" ]] || continue
      # map file dir -> serial_dir
      local serial_dir; serial_dir="$(dirname "$dir")"
      if [[ -z "${newest[$serial_dir]:-}" || "${ts%%.*}" -gt "${newest[$serial_dir]}" ]]; then
        newest[$serial_dir]="${ts%%.*}"
      fi
    done < <(find "$WATCH_DIR" -type f -iregex ".*\\.\\(${ext_regex_emacs}\\)$" -printf '%T@ %h\n' 2>/dev/null | sort -n)

    for serial_dir in "${!newest[@]}"; do
      [[ -d "$serial_dir" ]] || continue
      last=$(awk -F'\t' -v k="$serial_dir" '$1==k{print $2}' "$STATE" 2>/dev/null || true)
      if [[ "$RERUN_ON_CHANGE" != "true" && -n "$last" && "${newest[$serial_dir]}" -le "$last" ]]; then
        continue
      fi
      if wait_until_ready "$serial_dir"; then
        run_validate_once "$serial_dir"
        # update state
        tmp="${STATE}.tmp"; grep -v -F "$serial_dir"$'\t' "$STATE" 2>/dev/null >"$tmp" || true
        printf '%s\t%s\n' "$serial_dir" "${newest[$serial_dir]}" >>"$tmp"
        mv "$tmp" "$STATE"
      else
        echo "[Watcher] Timed out waiting for readiness: $serial_dir"
      fi
    done
    sleep "$POLL_INTERVAL"
  done
}

# --- Choose mode based on FS type & availability ---
mkdir -p "$WATCH_DIR"
fs_type="$(stat -f -c %T "$WATCH_DIR" 2>/dev/null || echo unknown)"
if [[ "$fs_type" == "nfs" || "$fs_type" == "nfs4" || "$fs_type" == "cifs" || "$fs_type" == "smbfs" ]]; then
  do_watch_polling
else
  if do_watch_inotify; then
    exit 0
  else
    do_watch_polling
  fi
fi

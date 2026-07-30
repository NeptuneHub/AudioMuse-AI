#!/usr/bin/env bash
# Capture and evaluate evidence for AMPACHE_TEST_PLAN.md around a full-library run.
#
#   ./ampache-evidence.sh start     # before you start the import
#   ./ampache-evidence.sh report    # after it finishes -> markdown for section 10
#
# Only judges what a NORMAL run can show. Cases needing fault injection (D7, D8, D14,
# D28a), a second provider (phase B/C) or a different config (D26, phase F) are listed
# as MANUAL and are never reported as passing.
#
# Config via env:
#   APACHE_ACCESS_LOG  Apache access log        (default /var/log/apache2/access.log)
#   AMPACHE_VHOST      vhost to filter to       (default music.lachlandewaard.org)
#   AMPACHE_APP_LOG    Ampache's own log        (default /var/log/ampache/ampache-music.log)
#   COMPOSE_DIR        AudioMuse compose dir    (default /mnt/music/audiomuse)
#   EVIDENCE_DIR       artefact output          (default ./ampache-evidence)
#
# The Apache log is shared by every vhost, so the baseline offset is counted on the RAW
# file and the vhost filter is applied afterwards. Filtering first would make the offset
# meaningless as soon as another vhost logged a request.
#
# Reading /var/log/apache2 usually needs root or the adm group: run with sudo -E (the -E
# keeps the env vars above) if 'start' reports it cannot read the log.
set -uo pipefail

ACCESS_LOG="${APACHE_ACCESS_LOG:-/var/log/apache2/access.log}"
VHOST="${AMPACHE_VHOST:-music.lachlandewaard.org}"
APP_LOG="${AMPACHE_APP_LOG:-/var/log/ampache/ampache-music.log}"
COMPOSE_DIR="${COMPOSE_DIR:-/mnt/music/audiomuse}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$PWD/ampache-evidence}"
STATE="$EVIDENCE_DIR/state.env"

die() { echo "ERROR: $*" >&2; exit 1; }

lines_of() { [ -r "$1" ] && wc -l < "$1" || echo 0; }

# Slice a log from a baseline line number, falling back to the whole file if it was
# rotated under us (a full-library import can easily outlive a daily rotation).
slice_from() { # file, baseline, outfile, label
  local file="$1" base="$2" out="$3" label="$4" now
  if [ ! -r "$file" ]; then : > "$out"; echo "  $label: not readable, skipped"; return; fi
  now=$(wc -l < "$file")
  if [ "$now" -lt "$base" ]; then
    cp "$file" "$out"
    echo "  $label: ROTATED during the run (was $base lines, now $now) - using the whole current file; check the rotated copy too"
  else
    tail -n "+$((base + 1))" "$file" > "$out"
    echo "  $label: $(wc -l < "$out") new lines"
  fi
}

cmd_start() {
  [ -r "$ACCESS_LOG" ] || die "cannot read $ACCESS_LOG (try: sudo -E $0 start)"
  mkdir -p "$EVIDENCE_DIR"
  {
    echo "ACCESS_LOG=$ACCESS_LOG"
    echo "ACCESS_BASE=$(lines_of "$ACCESS_LOG")"
    echo "APP_LOG=$APP_LOG"
    echo "APP_BASE=$(lines_of "$APP_LOG")"
    echo "VHOST=$VHOST"
    echo "START_TS=$(date --iso-8601=seconds)"
  } > "$STATE"

  # The image is what runs, not the source tree. Record what is actually loaded.
  ( cd "$COMPOSE_DIR" && docker compose exec -T audiomuse-ai-worker python3 -c "
import config, tasks.mediaserver.ampache as a
print('AMPACHE_PAGE_SIZE', config.AMPACHE_PAGE_SIZE, a._page_size())
print('LYRICS_ENABLED   ', config.LYRICS_ENABLED)
print('MUSIC_LIBRARIES  ', repr(config.MUSIC_LIBRARIES))
print('MEDIASERVER_TYPE ', config.MEDIASERVER_TYPE)
print('has_album_lyrics ', hasattr(a, '_album_lyrics'))
print('has_browse_ids   ', hasattr(a, 'get_album_track_ids'))
" ) > "$EVIDENCE_DIR/worker-config.txt" 2>&1 \
    || echo "(worker probe FAILED - is the stack up?)" >> "$EVIDENCE_DIR/worker-config.txt"

  ( cd "$COMPOSE_DIR" && git -C src rev-parse HEAD && git -C src status --short ) \
    > "$EVIDENCE_DIR/src-commit.txt" 2>&1

  echo "Baseline recorded in $EVIDENCE_DIR"
  cat "$EVIDENCE_DIR/worker-config.txt"
  echo "Start the import now, then run: $0 report"
}

verdict() { # label, already-evaluated status (0=pass), detail
  if [ "$2" -eq 0 ]; then echo "| $1 | PASS | $3 |"; else echo "| $1 | **CHECK** | $3 |"; fi
}

cmd_report() {
  [ -r "$STATE" ] || die "no baseline; run '$0 start' before the import"
  # shellcheck disable=SC1090
  . "$STATE"

  local raw="$EVIDENCE_DIR/access-raw.log"
  local slice="$EVIDENCE_DIR/access-during-run.log"
  local applog="$EVIDENCE_DIR/ampache-app-during-run.log"
  local wlog="$EVIDENCE_DIR/worker-during-run.log"

  echo "Slicing logs:"
  slice_from "$ACCESS_LOG" "$ACCESS_BASE" "$raw" "apache access"
  slice_from "$APP_LOG" "$APP_BASE" "$applog" "ampache app log"
  # Shared log: keep only this vhost's requests.
  grep -F "$VHOST" "$raw" > "$slice" || true
  echo "  vhost '$VHOST': $(wc -l < "$slice") of $(wc -l < "$raw") lines"
  ( cd "$COMPOSE_DIR" && docker compose logs --since "$START_TS" --no-color ) > "$wlog" 2>&1 || true

  [ -s "$slice" ] || die "no requests for vhost '$VHOST' in the window; wrong AMPACHE_VHOST, or the log format does not carry the vhost"

  local api="$EVIDENCE_DIR/api-actions.txt"
  grep -oE 'action=[a-z_]+' "$slice" | sort | uniq -c | sort -rn > "$api"

  local n_browse n_album_songs n_song n_ping n_handshake n_albums n_songs_action lyrics_off
  n_browse=$(grep -cE 'action=browse[^ ]*type=album' "$slice")
  n_album_songs=$(grep -cE 'action=album_songs' "$slice")
  n_song=$(grep -cE 'action=song&' "$slice")
  n_ping=$(grep -cE 'action=ping' "$slice")
  n_handshake=$(grep -cE 'action=handshake' "$slice")
  n_albums=$(grep -cE 'action=albums' "$slice")
  n_songs_action=$(grep -cE 'action=songs[&"]' "$slice")
  grep -qE 'LYRICS_ENABLED +False' "$EVIDENCE_DIR/worker-config.txt" && lyrics_off=0 || lyrics_off=1

  {
    echo "## Evidence for section 10 (generated $(date --iso-8601=seconds))"
    echo
    echo "Apache log \`$ACCESS_LOG\` from line $((ACCESS_BASE + 1)), vhost \`$VHOST\`, since \`$START_TS\`."
    echo
    echo "### What the worker actually had loaded"
    echo
    echo '```'
    cat "$EVIDENCE_DIR/worker-config.txt"
    echo "src commit: $(head -1 "$EVIDENCE_DIR/src-commit.txt")"
    echo '```'
    echo
    echo "### Ampache API calls observed"
    echo
    echo '```'
    cat "$api"
    echo '```'
    echo
    echo "### Log-observable cases"
    echo
    echo "| Case | Result | Evidence |"
    echo "| --- | --- | --- |"

    [ "$n_browse" -gt 0 ]; verdict "D28 dispatch loop uses browse" $? \
      "$n_browse browse(type=album) vs $n_album_songs album_songs; expect about one album_songs per album ANALYSED, not per album checked"
    if [ "$lyrics_off" -eq 0 ]; then
      echo "| D21/D27/D27a lyrics | N/A | LYRICS_ENABLED=false, so the lyrics stage never ran. Saw $n_song action=song. Zero here does NOT evidence D27: disabled and working are indistinguishable in a log |"
    else
      [ "$n_song" -eq 0 ]; verdict "D27 no per-track song refetch" $? "$n_song action=song calls with lyrics ENABLED"
    fi
    [ "$n_albums" -gt 0 ]; verdict "D12 album discovery ran" $? "$n_albums action=albums requests"
    if grep -qE 'action=albums[^ ]*cond=catalog' "$slice"; then
      echo "| D12a album filter server-side | PASS | \`cond=catalog,<id>\` present on the albums browse |"
    else
      echo "| D12a album filter server-side | N/A | no \`cond=catalog\` seen, expected when MUSIC_LIBRARIES is unset |"
    fi
    if [ "$n_songs_action" -gt 0 ]; then
      grep -qE 'action=songs[^ ]*cond=catalog' "$slice" \
        && echo "| D13 sweep filtered server-side | PASS | \`cond=catalog,<id>\` on the songs browse |" \
        || echo "| D13 sweep filtered server-side | N/A | sweep ran unfiltered (no MUSIC_LIBRARIES) |"
      echo "| D14b page size honoured | CHECK | limits seen: $(grep -oE 'action=songs[^ ]*limit=[0-9]+' "$slice" | grep -oE 'limit=[0-9]+' | sort -u | tr '\n' ' ') - compare with AMPACHE_PAGE_SIZE above |"
    else
      echo "| D13/D14b sweep | N/A | no \`action=songs\` in this window; analysis alone does not sweep |"
    fi
    if [ "$n_handshake" -eq 0 ] && [ "$n_ping" -gt 0 ]; then
      echo "| D24 API key via bearer header | PASS | $n_ping ping, 0 handshake |"
    elif [ "$n_handshake" -gt 0 ]; then
      echo "| D26 password mode | PASS | $n_handshake handshake, $n_ping ping - correct for a password. D24 needs an API-key run |"
    else
      echo "| D24/D26 auth mode | **CHECK** | neither ping nor handshake in the window |"
    fi
    ! grep -qE 'auth=[^&[:space:]]{8,}' "$slice"; verdict "D15 no secret in the query string" $? \
      "no long auth= value in the access log; also grep the worker log for the literal secret"
    ! grep -qE 'AMPACHE (CATALOGUE|ALBUM) FETCH FAILED|INCOMPLETE|RETURNED NO ALBUMS|REPORTED NO CATALOGUE' "$wlog"
    verdict "D12c/D14 no silent incompleteness" $? "no FAILED/INCOMPLETE/NO ALBUMS lines in the worker log"
    if [ -s "$applog" ]; then
      echo "| Ampache-side errors | CHECK | $(wc -l < "$applog") new lines in $APP_LOG, see ampache-app-during-run.log |"
    else
      echo "| Ampache-side errors | PASS | nothing new in $APP_LOG during the run |"
    fi

    echo
    echo "### Not covered by this run - do these deliberately"
    echo
    echo "| Case | Why a normal run cannot show it |"
    echo "| --- | --- |"
    echo "| D7, D8 | Need a session invalidated mid-analysis |"
    echo "| D14, D14a, D28a | Need an injected failure, a short \`total_count\`, or an ignored \`cond\` |"
    echo "| D25, D26 | Need a refused API key, and a password-configured run |"
    echo "| B*, C* | Need a second provider and a migration in both directions |"
    echo "| D17, D18, D19 | Playlist paths: run a Sonic Fingerprint / instant playlist |"
    echo "| F* | Only applies on a branch carrying LOCAL_FILE_ACCESS |"
  } > "$EVIDENCE_DIR/report.md"

  echo
  echo "Wrote $EVIDENCE_DIR/report.md"
  echo "Artefacts: $(cd "$EVIDENCE_DIR" && ls | tr '\n' ' ')"
  echo
  cat "$EVIDENCE_DIR/report.md"
}

case "${1:-}" in
  start)  cmd_start ;;
  report) cmd_report ;;
  *) echo "usage: $0 {start|report}"; exit 2 ;;
esac

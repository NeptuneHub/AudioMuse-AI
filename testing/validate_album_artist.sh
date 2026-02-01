#!/usr/bin/env bash
# ============================================================================
# validate_album_artist.sh
# ============================================================================
# Queries every AudioMuse test Postgres instance to verify that the
# album_artist column exists, is populated, and contains sensible data.
#
# Usage:
#   chmod +x validate_album_artist.sh
#   ./validate_album_artist.sh
#
# Requirements:
#   - psql (PostgreSQL client) installed on the host
#     Install:  sudo apt install postgresql-client   (Debian/Ubuntu)
#               brew install libpq                    (macOS)
#   - The AudioMuse test stacks must be running
# ============================================================================

set -euo pipefail

# --- Configuration ----------------------------------------------------------
DB_USER="audiomuse"
DB_PASS="audiomusepassword"
DB_NAME="audiomusedb"

declare -A INSTANCES=(
  ["Jellyfin"]=5433
  ["Emby"]=5434
  ["Navidrome"]=5435
  ["Lyrion"]=5436
)

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

PASS=0
FAIL=0
WARN=0

# --- Helper -----------------------------------------------------------------
run_query() {
  local port="$1"
  local query="$2"
  PGPASSWORD="$DB_PASS" psql -h localhost -p "$port" -U "$DB_USER" -d "$DB_NAME" \
    -t -A -c "$query" 2>/dev/null
}

separator() {
  echo ""
  echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ============================================================================
# Main
# ============================================================================

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       AudioMuse-AI  –  album_artist Validation Script          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

for provider in Jellyfin Emby Navidrome Lyrion; do
  port="${INSTANCES[$provider]}"
  separator
  echo -e "${BOLD}▶ ${provider}${NC}  (postgres localhost:${port})"
  echo ""

  # ---- 1. Connectivity check ------------------------------------------------
  if ! PGPASSWORD="$DB_PASS" psql -h localhost -p "$port" -U "$DB_USER" -d "$DB_NAME" \
       -c "SELECT 1" >/dev/null 2>&1; then
    echo -e "  ${RED}✗ FAIL${NC}  Cannot connect to Postgres on port ${port}"
    FAIL=$((FAIL + 1))
    continue
  fi
  echo -e "  ${GREEN}✓${NC}  Connected to Postgres"

  # ---- 2. Column existence ---------------------------------------------------
  col_exists=$(run_query "$port" \
    "SELECT COUNT(*) FROM information_schema.columns
     WHERE table_name = 'score' AND column_name = 'album_artist';")

  if [[ "$col_exists" -eq 1 ]]; then
    echo -e "  ${GREEN}✓${NC}  album_artist column exists in score table"
  else
    echo -e "  ${RED}✗ FAIL${NC}  album_artist column NOT found in score table"
    FAIL=$((FAIL + 1))
    continue
  fi

  # ---- 3. Row counts --------------------------------------------------------
  total=$(run_query "$port" "SELECT COUNT(*) FROM score;")
  echo -e "  ${CYAN}ℹ${NC}  Total tracks in score table: ${BOLD}${total}${NC}"

  if [[ "$total" -eq 0 ]]; then
    echo -e "  ${YELLOW}⚠ WARN${NC}  No tracks analysed yet – run analysis first"
    WARN=$((WARN + 1))
    continue
  fi

  populated=$(run_query "$port" \
    "SELECT COUNT(*) FROM score WHERE album_artist IS NOT NULL AND album_artist <> '';")
  empty=$(run_query "$port" \
    "SELECT COUNT(*) FROM score WHERE album_artist IS NULL OR album_artist = '';")

  echo -e "  ${CYAN}ℹ${NC}  album_artist populated: ${BOLD}${populated}${NC}"
  echo -e "  ${CYAN}ℹ${NC}  album_artist empty/null: ${BOLD}${empty}${NC}"

  if [[ "$populated" -gt 0 ]]; then
    pct=$(( populated * 100 / total ))
    echo -e "  ${GREEN}✓${NC}  Population rate: ${BOLD}${pct}%${NC}"
    PASS=$((PASS + 1))
  else
    echo -e "  ${RED}✗ FAIL${NC}  album_artist is entirely empty (0 populated rows)"
    FAIL=$((FAIL + 1))
  fi

  # ---- 4. Sample rows -------------------------------------------------------
  echo ""
  echo -e "  ${BOLD}Sample tracks (up to 10):${NC}"
  echo -e "  ─────────────────────────────────────────────────────────────────"
  PGPASSWORD="$DB_PASS" psql -h localhost -p "$port" -U "$DB_USER" -d "$DB_NAME" \
    -c "SELECT item_id, title, author, album, album_artist
        FROM score
        WHERE album_artist IS NOT NULL AND album_artist <> ''
        ORDER BY random()
        LIMIT 10;" 2>/dev/null | while IFS= read -r line; do
    echo "  $line"
  done

  # ---- 5. Distinct album_artist values --------------------------------------
  echo ""
  distinct=$(run_query "$port" \
    "SELECT COUNT(DISTINCT album_artist) FROM score WHERE album_artist IS NOT NULL AND album_artist <> '';")
  echo -e "  ${CYAN}ℹ${NC}  Distinct album_artist values: ${BOLD}${distinct}${NC}"

  # ---- 6. Check for 'Unknown' fallback dominance -----------------------------
  unknown_count=$(run_query "$port" \
    "SELECT COUNT(*) FROM score WHERE album_artist = 'Unknown';")
  if [[ "$total" -gt 0 && "$unknown_count" -gt 0 ]]; then
    unknown_pct=$(( unknown_count * 100 / total ))
    if [[ "$unknown_pct" -gt 50 ]]; then
      echo -e "  ${YELLOW}⚠ WARN${NC}  ${unknown_pct}% of tracks have album_artist = 'Unknown' – provider may not be returning the field"
      WARN=$((WARN + 1))
    else
      echo -e "  ${GREEN}✓${NC}  'Unknown' fallback: ${unknown_count} tracks (${unknown_pct}%) – acceptable"
    fi
  fi

  # ---- 7. Cross-check: album_artist vs author (artist) ----------------------
  mismatch=$(run_query "$port" \
    "SELECT COUNT(*) FROM score
     WHERE album_artist IS NOT NULL AND album_artist <> ''
       AND author IS NOT NULL AND author <> ''
       AND album_artist <> author;")
  echo -e "  ${CYAN}ℹ${NC}  Tracks where album_artist ≠ author (track artist): ${BOLD}${mismatch}${NC}"
  echo -e "      (non-zero is expected for compilations / VA albums)"

done

# ============================================================================
# Summary
# ============================================================================
separator
echo ""
echo -e "${BOLD}Summary${NC}"
echo -e "  Passed : ${GREEN}${PASS}${NC}"
echo -e "  Failed : ${RED}${FAIL}${NC}"
echo -e "  Warnings: ${YELLOW}${WARN}${NC}"
echo ""

if [[ "$FAIL" -gt 0 ]]; then
  echo -e "${RED}${BOLD}Some checks FAILED. Review output above.${NC}"
  exit 1
elif [[ "$WARN" -gt 0 ]]; then
  echo -e "${YELLOW}${BOLD}All checks passed with warnings.${NC}"
  exit 0
else
  echo -e "${GREEN}${BOLD}All checks PASSED.${NC}"
  exit 0
fi

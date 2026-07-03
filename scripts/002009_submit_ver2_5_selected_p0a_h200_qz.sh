#!/usr/bin/env sh
set -eu

# Submit the current selected Ver2.5 SourceSemanticMemory candidate.
#
# Decision source:
#   docs/ver2_5_debug_jobs_20260630.md
#
# Current selected candidate:
#   P0-A text-token source memory
#
# Explicitly not enabled by default:
#   P0-C codec_bottleneck first_4, because core48 showed higher repeat and
#   lower keep count than P0-A.
#
# Usage:
#   sh scripts/002009_submit_ver2_5_selected_p0a_h200_qz.sh
#
# Optional smoke:
#   DRY_RUN=1 sh scripts/002009_submit_ver2_5_selected_p0a_h200_qz.sh

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
ROOT=$(CDPATH= cd "$SCRIPT_DIR/.." && pwd)

EXPERIMENT_SCRIPT="${EXPERIMENT_SCRIPT:-$ROOT/scripts/004029_submit_ver2_5_source_semantic_experiments.sh}"
EXPERIMENT="${EXPERIMENT:-ver2_5_selected_p0a_text_token_memory_5k}"
DRY_RUN="${DRY_RUN:-0}"

if [ ! -f "$EXPERIMENT_SCRIPT" ]; then
  echo "ERROR: missing experiment script: $EXPERIMENT_SCRIPT" >&2
  exit 1
fi

echo "[ver2.5-selected] experiment=$EXPERIMENT"
echo "[ver2.5-selected] root=$ROOT"
echo "[ver2.5-selected] dry_run=$DRY_RUN"
echo "[ver2.5-selected] memory=text_tokens"
echo "[ver2.5-selected] disabled_memory=codec_bottleneck"

if [ "$DRY_RUN" = "1" ]; then
  sh "$EXPERIMENT_SCRIPT" "$EXPERIMENT" --dry-run
else
  sh "$EXPERIMENT_SCRIPT" "$EXPERIMENT"
fi

#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Interactive, user-operated image builder; does not deploy to a cluster.
set -euo pipefail

usage() {
  cat <<'HELP'
Usage: ./deploy.sh <version>
Example: ./deploy.sh 0.13.0

Select CPU and/or CUDA 12.4, then push to GHCR or load locally.
All builds target linux/amd64. Python 3.11+, Git and Docker Buildx are required.
Push requires clean HEAD equal to freshly fetched canonical origin/main.
Existing release tags are never overwritten; deselect published variants to resume.
DOCKER_CONTEXT defaults to desktop-linux. Existing Docker authentication may be
used, or choose login with GHCR_USERNAME/GHCR_TOKEN or hidden token input.
No Git tags, GitHub Releases, platform pins or cluster deployments are created.
HELP
}

if [[ "${1:-}" == --help ]] && (( $# == 1 )); then
  usage
  exit 0
fi
if (( $# != 1 )); then
  usage >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")"
VERSION="$1"
SOURCE_URL=https://github.com/neurwerk/k8s_stack_pii_engine
IMAGE=ghcr.io/neurwerk/k8s-stack-pii-engine
docker_cmd=(docker --context "${DOCKER_CONTEXT:-desktop-linux}")

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
command -v python3 >/dev/null || fail 'Python 3.11+ is required.'
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
  || fail 'Python 3.11+ is required.'
python3 scripts/verify_release_tag.py "v${VERSION}"
REVISION="$(git rev-parse HEAD)"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]] || fail 'Invalid source commit.'

variants=()
selected_count=0
read -r -p 'Build CPU image? (Y/n): ' answer
if [[ ! "$answer" =~ ^[Nn]$ ]]; then
  variants+=(cpu)
  selected_count=$((selected_count + 1))
fi
read -r -p 'Build CUDA 12.4 image? (Y/n): ' answer
if [[ ! "$answer" =~ ^[Nn]$ ]]; then
  variants+=(cu124)
  selected_count=$((selected_count + 1))
fi
(( selected_count > 0 )) || fail 'No image selected.'
output_flag=--push
read -r -p 'Push selected images to GHCR? (Y/n): ' answer
[[ "$answer" =~ ^[Nn]$ ]] && output_flag=--load
printf '\nVariants: %s\nPlatform: linux/amd64\nOutput: %s\nSource: %s@%s\n' \
  "${variants[*]}" "$output_flag" "$SOURCE_URL" "$REVISION"
read -r -p 'Proceed? (y/N): ' answer
if [[ ! "$answer" =~ ^[Yy]$ ]]; then
  printf 'Cancelled.\n'
  exit 0
fi

[[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
  || fail 'Git worktree must be clean before building.'
if [[ "$output_flag" == --push ]]; then
  case "$(git remote get-url origin)" in
    "$SOURCE_URL"|"$SOURCE_URL.git"|git@github.com:neurwerk/k8s_stack_pii_engine.git) ;;
    *) fail 'origin must be the canonical PII Engine repository.' ;;
  esac
  git fetch --quiet --no-tags origin refs/heads/main:refs/remotes/origin/main
  [[ "$REVISION" == "$(git rev-parse refs/remotes/origin/main)" ]] \
    || fail 'HEAD must equal fetched canonical origin/main.'
fi
"${docker_cmd[@]}" info >/dev/null || fail 'Docker is not running in the selected context.'

if [[ "$output_flag" == --push ]]; then
  read -r -p 'Log in to GHCR now? (y/N; otherwise use existing Docker auth): ' answer
  if [[ "$answer" =~ ^[Yy]$ ]]; then
    ghcr_username="${GHCR_USERNAME:-}"
    ghcr_token="${GHCR_TOKEN:-}"
    [[ -n "$ghcr_username" ]] || read -r -p 'GHCR username: ' ghcr_username
    if [[ -z "$ghcr_token" ]]; then
      read -r -s -p 'GHCR token with write:packages permission: ' ghcr_token
      printf '\n'
    fi
    [[ -n "$ghcr_username" && -n "$ghcr_token" ]] || fail 'Username and token are required.'
    printf '%s' "$ghcr_token" | "${docker_cmd[@]}" login ghcr.io \
      --username "$ghcr_username" --password-stdin
    unset ghcr_token GHCR_TOKEN
  fi
fi

build_dir="$(mktemp -d "${TMPDIR:-/tmp}/pii-engine-build.XXXXXX")"
trap 'rm -rf "$build_dir"' EXIT
if [[ "$output_flag" == --push ]]; then
  printf 'Checking every selected immutable tag before building ...\n'
  selected_tags=()
  for variant in "${variants[@]}"; do
    selected_tags+=("${VERSION}-${variant}")
  done
  python3 scripts/ghcr_preflight.py "${selected_tags[@]}"
fi

# Build only tracked committed files, never ignored backups or workstation data.
mkdir "$build_dir/context"
git archive "$REVISION" | tar -x -C "$build_dir/context"
for variant in "${variants[@]}"; do
  ref="${IMAGE}:${VERSION}-${variant}"
  printf '\nBuilding %s for linux/amd64 ...\n' "$ref"
  "${docker_cmd[@]}" buildx build --platform linux/amd64 \
    --build-arg "ACCELERATOR=${variant}" --tag "$ref" \
    --label "org.opencontainers.image.source=${SOURCE_URL}" \
    --label "org.opencontainers.image.revision=${REVISION}" \
    --label "org.opencontainers.image.version=${VERSION}-${variant}" \
    --file "$build_dir/context/Dockerfile" \
    --metadata-file "$build_dir/${variant}.json" "$output_flag" "$build_dir/context"
  digest="$(python3 - "$build_dir/${variant}.json" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as metadata_file:
    digest = json.load(metadata_file).get("containerimage.digest", "")
if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
    raise SystemExit("Buildx metadata did not contain a valid image digest")
print(digest)
PY
  )"
  printf 'Result (%s): %s@%s\n' "$output_flag" "$ref" "$digest"
done
printf '\nCompleted. No platform pins or cluster resources changed.\n'

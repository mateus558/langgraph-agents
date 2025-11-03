#!/usr/bin/env bash
set -euo pipefail

# Remote deployment settings (override with env vars if needed)
DEPLOY_HOST=${DEPLOY_HOST:-192.168.30.100}
DEPLOY_USER=${DEPLOY_USER:-mateuscmarim}
DEPLOY_PATH=${DEPLOY_PATH:-/home/mateuscmarim/langgraph-agents}
# Extra SSH options (e.g., -p 22 -i ~/.ssh/id_rsa)
SSH_OPTS=${SSH_OPTS:-}

"${DEBUG:-false}" && set -x || true

# Any arguments passed to this script are forwarded to `docker compose`
DOCKER_ARGS=${*:-}

generate_architecture() {
	echo "Generating ARCHITECTURE.md..."
	if command -v python >/dev/null 2>&1; then
		python scripts/generate_architecture_md.py
	elif command -v uv >/dev/null 2>&1; then
		uv run python scripts/generate_architecture_md.py
	else
		echo "Error: neither 'python' nor 'uv' found in PATH; cannot generate ARCHITECTURE.md" >&2
		exit 1
	fi
}

echo "Generating Dockerfile with langgraph..."
# Generate Dockerfile (overwrites or creates Dockerfile in repo root)
uv run langgraph dockerfile Dockerfile

# Always regenerate architecture docs before committing/deploying
generate_architecture

# Commit & push if Dockerfile or ARCHITECTURE.md changed
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	# Determine if tracked or content changed for files of interest
	FILES=("Dockerfile" "ARCHITECTURE.md")
	SHOULD_COMMIT=0
	for f in "${FILES[@]}"; do
		if [ -f "$f" ]; then
			if ! git diff --quiet -- "$f" || ! git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
				SHOULD_COMMIT=1
			fi
		fi
	done

	if [ "$SHOULD_COMMIT" -eq 1 ]; then
		echo "Staging files: ${FILES[*]}"
		git add ${FILES[*]} 2>/dev/null || true
		# If there's nothing to commit git commit will fail; guard that
		if git diff --cached --quiet; then
			echo "No staged changes to commit."
		else
			git commit -m "Docs: update ARCHITECTURE.md; chore: update Dockerfile"
			echo "Pushing commit to remote..."
			git push
		fi
	else
		echo "No changes in Dockerfile or ARCHITECTURE.md; skipping commit/push."
	fi
else
	echo "Not a git repo; skipping commit/push step."
fi

echo "Deploying to ${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH}"

ssh ${SSH_OPTS} "${DEPLOY_USER}@${DEPLOY_HOST}" \
	"set -euo pipefail; cd '${DEPLOY_PATH}' && docker compose up -d --build ${DOCKER_ARGS}"

echo "Deployment command executed on remote host."

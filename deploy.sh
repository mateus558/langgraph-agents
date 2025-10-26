#!/usr/bin/env bash
set -euo pipefail

# Remote deployment settings (override with env vars if needed)
DEPLOY_HOST=${DEPLOY_HOST:-192.168.30.100}
DEPLOY_USER=${DEPLOY_USER:-mateuscmarim}
DEPLOY_PATH=${DEPLOY_PATH:-/home/mateuscmarim/ai-server}
# Extra SSH options (e.g., -p 22 -i ~/.ssh/id_rsa)
SSH_OPTS=${SSH_OPTS:-}

# Any arguments passed to this script are forwarded to `docker compose`
DOCKER_ARGS=${*:-}

echo "Generating Dockerfile with langgraph..."
# Generate Dockerfile (overwrites or creates Dockerfile in repo root)
langgraph dockerfile Dockerfile

# Commit & push only if Dockerfile changed
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
	# Add the file and commit if there are changes
	if ! git diff --quiet -- Dockerfile || ! git ls-files --error-unmatch Dockerfile >/dev/null 2>&1; then
		echo "Staging Dockerfile and committing..."
		git add Dockerfile
		# If there's nothing to commit git commit will fail; guard that
		if git diff --cached --quiet; then
			echo "No staged changes to commit."
		else
			git commit -m "Dockerfile updated"
			echo "Pushing commit to remote..."
			git push
		fi
	else
		echo "Dockerfile unchanged; skipping commit/push."
	fi
else
	echo "Not a git repo; skipping commit/push step."
fi

echo "Deploying to ${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH}"

ssh ${SSH_OPTS} "${DEPLOY_USER}@${DEPLOY_HOST}" \
	"set -euo pipefail; cd '${DEPLOY_PATH}' && docker compose up -d --build ${DOCKER_ARGS}"

echo "Deployment command executed on remote host."


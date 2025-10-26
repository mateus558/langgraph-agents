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

echo "Deploying to ${DEPLOY_USER}@${DEPLOY_HOST}:${DEPLOY_PATH}"

ssh ${SSH_OPTS} "${DEPLOY_USER}@${DEPLOY_HOST}" \
	"set -euo pipefail; cd '${DEPLOY_PATH}' && docker compose up -d --build ${DOCKER_ARGS}"

echo "Deployment command executed on remote host."


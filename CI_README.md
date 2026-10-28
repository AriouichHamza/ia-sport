# GitLab CI — POC (README)

This document explains the GitLab CI pipeline added at `.gitlab-ci.yml` in this repository, the required CI/CD variables, and deployment notes for the project's Gradio app and Weaviate knowledge base.

Summary
- Pipeline stages: `lint`, `test`, `build`, `deploy`.
- The `build` stage builds a Docker image and pushes it to the project's Container Registry.
- The `deploy` stage (manual) SSHes to a host, pulls the image and runs it as a container.

Required GitLab CI variables (set in GitLab → Settings → CI/CD → Variables)
- `CI_REGISTRY` (provided by GitLab for container registry)
- `CI_REGISTRY_USER` (provided by GitLab)
- `CI_REGISTRY_PASSWORD` (provided by GitLab)
- `DEPLOY_SSH_KEY` — private SSH key for the deployment user (protect & mask)
- `DEPLOY_SSH_USER` — username on the deployment host
- `DEPLOY_SSH_HOST` — deployment host (IP or domain)
- `DEPLOY_PORT` — optional; port on remote host to map Gradio (defaults to `80` in CI)

Notes about Gradio app
- Your app (the Gradio UI) listens on port `7860` by default. The CI deploy job runs the container with `-p ${DEPLOY_PORT:-80}:7860` so the remote host port (DEPLOY_PORT) will be exposed to the internet.
- If you want a different mapping (or to use a reverse proxy such as Nginx), update the `docker run` line in `.gitlab-ci.yml` or manage routing on the host.

Notes about Weaviate
- The repository includes `modules/run_weaviate.sh` which attempts to start a Weaviate instance (via Docker). The pipeline does not start Weaviate automatically on the remote host.
- Options for production:
  - Run Weaviate on the same remote host (Docker) and ensure the container/port are accessible to the app.
  - Use a managed Weaviate instance and set `WEAVIATE_URL` environment variable in your deployment/run command.
  - If you wish the CI to start Weaviate on the remote host, modify the `deploy` job to execute `modules/run_weaviate.sh` or run a `docker compose` stack there.

Testing locally
- Build and run the Docker image locally (from `POC` root):
```bash
docker build -t poc_local:latest .
docker run -p 7860:7860 --name poc_local -d poc_local:latest
# open http://localhost:7860
```

- To simulate the remote deploy flow (manual):
  - Copy your SSH public key to the remote host's `~/.ssh/authorized_keys` for the `DEPLOY_SSH_USER`.
  - Ensure Docker is installed and the user can run `docker pull` and `docker run`.

Security & runner notes
- Protect `DEPLOY_SSH_KEY` and mark it as `protected` if you only want it available on protected branches/tags.
- The `build` job uses Docker-in-Docker (dind). Ensure you have a GitLab runner with Docker privileged mode enabled for that job.
- If you don't want to use dind, build images in CI using a remote builder or prebuilt images.

Common changes you might want
- Switch remote deploy to `docker-compose` — update `.gitlab-ci.yml` `deploy` job to copy a `docker-compose.yml` and run `docker compose up -d` on the host.
- Add health-checks in `deploy` that curl the Gradio endpoint after deployment and fail if not healthy.
- Add artifact collection (logs) or version tagging (use `CI_COMMIT_TAG` for production releases).

Next steps I applied
- Added `.gitlab-ci.yml` to `POC` with a lint/test/build/deploy flow.
- Created this `CI_README.md` documenting variables and deployment notes.

If you'd like, I can:
- Update the `deploy` job to use `docker-compose` on the remote host.
- Add a small `docker-compose.yml` template to run both the Gradio app and Weaviate together.
- Add a short GitLab README snippet showing how to set variables (copyable).

Env file and example
- A sample env file for the UI and deployment has been added at `POC/ui/.env.example`.
- Copy that file to `POC/ui/.env` on the host or provide equivalent variables via GitLab CI variables.
- Example variables in `POC/ui/.env.example`:
  - `OPENAI_API_KEY` — your OpenAI key (required by KnowledgeBase ingestion/embeddings).
  - `WEAVIATE_URL` — URL to the Weaviate instance (default `http://weaviate:8080` when using the included compose service).
  - `PORT` — internal Gradio port (7860 by default).
  - `DEPLOY_PORT` — external host port to expose Gradio (mapped in the CI deploy to the `.env` on the host).

Security reminder
- Never commit a `.env` with secrets. Use GitLab CI variables (`OPENAI_API_KEY`, `DEPLOY_SSH_KEY`, etc.) for secure deploys.


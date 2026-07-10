# Playlist Curator Isolated Test Deployment Design

## Goal

Address the three unresolved Bandit findings on PR 417, then deploy the branch as a new local test instance without sharing the production worker's queue, database, network, volumes, ports, or GPU.

## Current Environment

- The production worker runs as Compose project `audiomuse-ai-worker` in container `audiomuse-ai-worker-instance`.
- It connects to external PostgreSQL and Redis at `192.168.1.172`.
- It reserves the only GPU, device 0, which already uses most of its memory.
- The stopped `audiomuse-test` project owns PostgreSQL volume `audiomuse-test_test-postgres-data`.
- PR 417 has three unresolved Bandit B608 findings at the dynamic smart-search query executions.

## Security Finding Resolution

The filter query builder constructs SQL structure only from fixed field, operator, and join allowlists. Request values are passed separately as psycopg2 parameters. The Bandit findings are therefore false positives at the three final query assembly sites.

The implementation will:

1. Add regression tests using hostile field, operator, match-mode, and value strings.
2. Prove that unrecognized structural inputs cannot enter the SQL clause.
3. Prove that recognized filter values remain in the parameter list and not the SQL string.
4. Add narrowly scoped `# nosec B608` annotations with an explanatory comment at the three flagged executions.
5. Run the focused security tests, Ruff, Bandit, and the relevant unit suite before committing.
6. Push the fix and wait for PR checks to complete before deployment.

No GitHub review thread will be replied to or resolved automatically. A successful code-scanning rerun may mark the findings outdated; any remaining thread state will be reported.

## Test Deployment Architecture

The new Compose project will be named `playlist-curator-test` and will contain:

- `postgres`: PostgreSQL 15 using a cloned test-data volume.
- `redis`: Redis 7 using a new empty volume.
- `audiomuse-ai-flask`: a CPU-only image built from the current branch.

No test worker will be started. Playlist Curator search, extension, duplicate review, streaming, and playlist-save routes run in the Flask service and can use the index stored in the cloned database. Excluding a worker removes all risk of consuming production jobs and avoids CPU or GPU competition from background analysis.

## Data Isolation

The source volume `audiomuse-test_test-postgres-data` will be mounted read-only in a short-lived helper container and copied into a newly created `playlist-curator-test_postgres-data` volume. The source volume remains unchanged.

Redis will always start from a new `playlist-curator-test_redis-data` volume. Stale jobs from the stopped test stack will not be copied.

The test services will use only these internal endpoints:

- `POSTGRES_HOST=postgres`
- `POSTGRES_PORT=5432`
- `REDIS_URL=redis://redis:6379/0`

PostgreSQL and Redis will not publish host ports.

## Runtime and Resource Isolation

- Compose project: `playlist-curator-test`
- Dedicated network: `playlist-curator-test_default`
- Flask host port: `18001`
- Image: `audiomuse-ai-playlist-curator:test`
- Dockerfile default CPU base: `ubuntu:24.04`
- No NVIDIA device request
- `NVIDIA_VISIBLE_DEVICES=none`
- `USE_GPU_CLUSTERING=false`

The stopped `audiomuse-test` Flask configuration will provide media-server settings. Values will be copied into the deployment process environment in memory and will not be written to a repository or temporary environment file.

## Deployment Sequence

1. Record the production worker container ID, start time, network, Redis/PostgreSQL endpoints, and GPU reservation.
2. Verify port `18001` and the planned container and volume names are unused.
3. Build the CPU-only branch image.
4. Create new test volumes and clone the stopped test PostgreSQL data read-only.
5. Generate a temporary Compose file outside the repository.
6. Start only PostgreSQL, Redis, and Flask.
7. Wait for PostgreSQL readiness and the Flask health endpoint.
8. Verify the Playlist Curator pages and API respond.
9. Reinspect the production worker and prove its identity, start time, network, endpoints, and GPU reservation are unchanged.

## Failure Handling

- If the security checks fail, deployment stops before any containers are created.
- If the image build fails, existing containers and volumes remain untouched.
- If the database clone fails, the new partial volume is removed only after its resolved name is verified.
- If a test service fails health checks, only the `playlist-curator-test` project is stopped; production is never targeted by a Compose command.
- The source `audiomuse-test` volume and all `audiomuse-ai-worker` resources are never removed or modified.

## Verification Criteria

Deployment is successful only when:

- PR 417 has no failing required checks after the Bandit fix.
- The three test services are running on the dedicated test network.
- Flask responds on `http://localhost:18001`.
- Test Flask resolves PostgreSQL and Redis by the internal service names.
- No test container is attached to `audiomuse-ai-worker_default`.
- No test container has an NVIDIA device request.
- The production worker container ID and start time are unchanged.
- The production worker remains running with its original external Redis/PostgreSQL endpoints.

## Non-Goals

- Starting a test RQ worker.
- Reanalyzing the test library.
- Publishing PostgreSQL or Redis to the host.
- Modifying production media, database, Redis, containers, volumes, or networks.
- Replying to or resolving GitHub review threads without explicit user authorization.

# PII Engine

PII Engine detects personal information in text and applies configured safety rules, transformations, blocking, and routing decisions. It serves the gateway external processor and Studio through separate workload-authorized API routes.

The Engine owns detection and policy evaluation. The [gateway processor](https://github.com/neurwerk/k8s_stack_agentgateway_extproc) owns traffic adaptation and response restoration. The [Base chart](https://github.com/neurwerk/k8s_stack_base/tree/main/charts/pii-engine) owns deployment, mounted configuration, dependency wiring, and image selection.

- [Architecture](architecture/README.md): request flow, detection layers, state, and failure behavior.
- [Development](development/README.md): local prerequisites, checks, and image build.
- [Operations](operations/README.md): configuration, health checks, and recovery ownership.

See the shared [platform architecture](https://github.com/neurwerk/docs/blob/main/dev/architecture/README.md) and [authentication guide](https://github.com/neurwerk/docs/blob/main/dev/authentication/README.md) for gateway and user-access boundaries.

<!--
SYNC IMPACT REPORT
==================
Version Change  : (unset template placeholders) → 1.0.0
Bump Type       : MINOR — first concrete population of all principles and sections
                  (template-to-constitution promotion treated as minor, no prior version existed)

Modified Principles (old placeholder → new title):
  [PRINCIPLE_1_NAME] → I. Hexagonal Architecture (Ports & Adapters)
  [PRINCIPLE_2_NAME] → II. SOLID Principles
  [PRINCIPLE_3_NAME] → III. Design Patterns
  [PRINCIPLE_4_NAME] → IV. English-Only Codebase
  [PRINCIPLE_5_NAME] → V. Clean Code & Good Development Practices

Added Sections:
  - Technology Stack & Quality Standards  (replaces [SECTION_2_NAME])
  - Development Workflow & Quality Gates  (replaces [SECTION_3_NAME])
  - Governance                            (filled [GOVERNANCE_RULES])

Removed Sections: None

Templates Updated:
  ✅ .specify/templates/plan-template.md  — Constitution Check gates populated
  ✅ .specify/templates/tasks-template.md — Hexagonal-layer task guidance added
  ✅ .specify/templates/spec-template.md  — Architecture compliance note added
  ✅ .specify/memory/constitution.md      — This file (primary artifact)

Deferred TODOs: None — all placeholders resolved.
-->

# p2c-mcts-core Constitution

## Core Principles

### I. Hexagonal Architecture (Ports & Adapters)

All business logic MUST reside in the **domain core** (entities, use cases, domain services).
Infrastructure concerns — HTTP frameworks, databases, message brokers, external APIs — MUST be
implemented as **adapters** behind **port interfaces** defined by the domain.

- The domain core MUST NOT import any infrastructure framework or library directly.
- Every external dependency MUST be reached through a port (abstract class / protocol) defined
  inside the domain layer.
- Inbound adapters (REST controllers, CLI, event consumers) MUST delegate to use cases without
  containing business logic.
- Outbound adapters (repositories, HTTP clients, caches) MUST implement domain-defined port
  interfaces.
- The directory structure MUST reflect the layering:
  `src/core/` (domain), `src/ports/` (interfaces), `src/adapters/` (implementations).
- Tests for the domain core MUST run using fakes/stubs for all ports; no real infrastructure
  is permitted inside unit tests.

**Rationale**: Decoupling business logic from infrastructure makes the MCTS/MDP core independently
testable, swappable, and resilient to technology changes — critical for a service whose computation
engine and external integrations (game data, caching, persistence) evolve at different rates.

### II. SOLID Principles (NON-NEGOTIABLE)

All code MUST adhere to the five SOLID principles:

- **Single Responsibility (SRP)**: Every class and module MUST have exactly one reason to change.
  Classes that mix orchestration, business rules, and I/O are forbidden.
- **Open/Closed (OCP)**: Modules MUST be open for extension and closed for modification.
  New behavior MUST be introduced via new classes or injected strategies, not by editing
  existing logic paths.
- **Liskov Substitution (LSP)**: Every adapter implementing a port MUST be fully substitutable
  for that port without altering the correctness of the use case. Implementations MUST NOT
  tighten preconditions or weaken postconditions defined by the port.
- **Interface Segregation (ISP)**: Interfaces (ports) MUST be narrow and role-specific.
  Callers MUST NOT be forced to depend on methods they do not use. Broad "catch-all"
  interfaces are forbidden.
- **Dependency Inversion (DIP)**: High-level modules (use cases) MUST depend on abstractions
  (ports), never on concrete adapter classes. Dependency injection MUST be used to wire
  adapters into the application at composition-root level.

**Rationale**: SOLID principles directly reinforce the hexagonal architecture and keep each
MCTS subsystem (selection, expansion, simulation, backpropagation) and each MDP component
(state, action, transition, reward) independently maintainable and testable.

### III. Design Patterns

Appropriate **GoF (Gang of Four)** and **enterprise patterns** MUST be applied where they
reduce complexity, increase reusability, or communicate intent clearly.

- Patterns MUST be chosen based on the problem; applying a pattern for its own sake is
  forbidden and constitutes unnecessary complexity.
- Every non-trivial pattern choice MUST be documented with an inline comment referencing the
  pattern name and the problem it solves, or an ADR under `docs/adr/`.
- Preferred patterns for this project (non-exhaustive):
  - **Strategy** — interchangeable MCTS policies (UCB1, selection, simulation, rollout).
  - **Factory / Abstract Factory** — constructing MCTS nodes and crafting-action objects.
  - **Repository** — abstracting game-state and knowledge-base persistence.
  - **Command** — encapsulating crafting actions as first-class executable objects.
  - **Observer** — propagating state-change events across domain subsystems.
  - **Template Method** — enforcing the four-phase MCTS loop while allowing per-phase
    customization.
- Explicit anti-patterns that are **prohibited**: God Class, Singleton overuse, Anemic Domain
  Model, Service Locator.

**Rationale**: Explicit pattern application makes algorithmic intent legible to new contributors
and ensures the complex MCTS/MDP machinery is decomposed into cohesive, independently testable
units with well-known extension points.

### IV. English-Only Codebase (NON-NEGOTIABLE)

All artifacts in this repository MUST be written exclusively in English.

- Source code (identifiers, comments, docstrings, type annotations) MUST be in English.
- Documentation (README, ADRs, specs, plans, API docs, wikis) MUST be in English.
- Git commit messages and pull request titles/descriptions MUST be in English.
- API response payloads, error messages, and log entries MUST be in English.
- Non-English text in any of the above is a **merge blocker** and MUST be corrected before
  the pull request can be approved.

**Rationale**: The project targets an international developer community and open-source
collaboration. A single language removes ambiguity, simplifies tooling (search, static
analysis, documentation generation), and guarantees every contributor can review every
artifact without translation.

### V. Clean Code & Good Development Practices

All code MUST meet the following baseline quality standards:

- **Naming**: Identifiers MUST be descriptive, unambiguous, and follow the language's idiomatic
  casing conventions. Abbreviations MUST be avoided unless they are universally understood
  domain terms (e.g., `mcts`, `mdp`, `ucb`, `uct`).
- **Functions / Methods**: Each MUST do exactly one thing. Functions exceeding 30 logical lines
  SHOULD be refactored. Parameter lists exceeding 4 items SHOULD be grouped into a value object
  or dataclass.
- **Testing**: All public use-case ports and domain services MUST have unit tests. Integration
  tests MUST cover the primary API contract. Domain-core test coverage MUST NOT fall below 80%.
- **Documentation**: Every public module, class, and function MUST have a docstring stating
  its purpose, parameters, and return value. Architecture decisions MUST be captured as ADRs
  under `docs/adr/`.
- **Error Handling**: Errors MUST be typed, domain-meaningful, and propagated cleanly through
  port boundaries. Bare `except Exception` catches and silent failures are forbidden.
- **No Dead Code**: Commented-out code MUST NOT be committed. Unused imports, variables, and
  unreachable branches are forbidden and enforced by the linter.
- **Immutability Preference**: Value objects and domain entities SHOULD be immutable where
  feasible, using frozen dataclasses or equivalent constructs.

**Rationale**: Clean-code practices are the operational floor that makes all other principles
achievable. Without them, hexagonal layers collapse into tangled dependencies and SOLID
adherence becomes superficial.

## Technology Stack & Quality Standards

- **Language**: Python 3.11+
- **HTTP Framework**: FastAPI (inbound adapter only; MUST NOT be imported in `src/core/`)
- **Data Validation**: Pydantic v2 (used in the API adapter layer; domain uses plain dataclasses
  or value objects)
- **Core Computation**: NumPy (numerical operations), NetworkX (tree/graph structures)
- **Testing**: pytest + pytest-cov; port fakes/stubs for unit tests; `httpx` for API integration
  tests
- **Static Analysis**: Mypy strict mode (type checking); Ruff (linting + formatting)
- **Containerization**: Docker + Docker Compose for local development and deployment
- **Optional Adapters**: Redis (game-state caching), PostgreSQL (crafting knowledge base)

CI quality gates — all MUST pass before merge:

| Gate | Command | Threshold |
|------|---------|-----------|
| Unit + integration tests | `pytest` | 0 failures |
| Type checking | `mypy --strict src/` | 0 errors |
| Linting | `ruff check src/ tests/` | 0 errors |
| Domain-core coverage | `pytest --cov=src/core` | ≥ 80% |
| Architecture guard | `import-linter` or `forbidden-imports` check | 0 violations |

## Development Workflow & Quality Gates

1. **Branch strategy**: Feature branches from `main` using the convention
   `[type]/[short-description]` (e.g., `feature/mcts-ucb1-policy`, `fix/reward-normalization`,
   `refactor/state-value-object`).
2. **Commit messages**: Conventional Commits format in English:
   `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `perf:`.
3. **Pull requests**: MUST reference the relevant spec or task ID in the description; MUST pass
   all CI quality gates; MUST receive at least one peer review and explicit approval.
4. **Constitution Check** (mandatory on every PR and feature plan):
   - [ ] Domain core (`src/core/`) contains zero direct imports of infrastructure libraries
   - [ ] Every new external dependency is accessed through a port defined in `src/ports/`
   - [ ] Each new class/module has a single, clearly stated responsibility (SRP)
   - [ ] New behavior is introduced via extension, not modification of existing logic (OCP)
   - [ ] All port interfaces are narrow and role-specific; no forced unused-method dependencies (ISP)
   - [ ] Use-case classes receive all adapters via dependency injection; no `ConcreteAdapter()` calls
         inside `src/core/` (DIP)
   - [ ] Every applied design pattern is documented (inline comment or ADR reference)
   - [ ] All code, comments, docstrings, commit messages, and PR descriptions are in English
   - [ ] Domain-core test coverage is ≥ 80%
   - [ ] No dead code, no commented-out blocks committed
5. **Architecture review**: Any proposal to add a new adapter, redefine a port, or introduce
   a cross-layer dependency MUST be captured in an ADR (`docs/adr/`) before implementation.

## Governance

This Constitution is the supreme governing document for the `p2c-mcts-core` project.
All other guidelines, team preferences, and conventions are subordinate to it.

- **Amendment procedure**: Amendments require a dedicated pull request with title prefix
  `chore: amend constitution to vX.Y.Z`, a written rationale explaining the change and its
  impact, and approval from at least one project maintainer. The `LAST_AMENDED_DATE` MUST be
  updated on every amendment.
- **Versioning policy**:
  - MAJOR: Backward-incompatible principle removal or fundamental redefinition.
  - MINOR: New principle or section added, or materially expanded/tightened guidance.
  - PATCH: Clarifications, wording corrections, or non-semantic refinements.
- **Compliance review**: Constitution compliance MUST be verified by every pull-request reviewer
  using the "Constitution Check" checklist in the Development Workflow section above.
- **Conflict resolution**: When a spec, plan, or task conflicts with this Constitution, the
  Constitution wins. Conflicts MUST be surfaced explicitly in the PR or design review, not
  silently resolved or ignored.
- **Guidance files**: Use `README.md` for project overview and quick-start guidance; use
  `docs/adr/` for architecture decision records that reference constitution principles.

**Version**: 1.0.0 | **Ratified**: 2026-03-11 | **Last Amended**: 2026-03-11

# Project conventions

This is the contributor reference for how KinoMLX code, APIs, documentation, and
validation are maintained. [Project structure](STRUCTURE.md) maps the implementation;
the [user guides](../README.md#documentation) explain how to run it.

- [Documentation](#documentation)
- [Architecture and ownership](#architecture-and-ownership)
- [Python style, imports, and types](#python-style-imports-and-types)
- [Configuration and command boundaries](#configuration-and-command-boundaries)
- [Weights and adapters](#weights-and-adapters)
- [Precision and numerical correctness](#precision-and-numerical-correctness)
- [MLX, native media, and memory](#mlx-native-media-and-memory)
- [Reporting and profiling](#reporting-and-profiling)
- [Dependencies and provenance](#dependencies-and-provenance)
- [Tests and validation](#tests-and-validation)
- [Performance measurements](#performance-measurements)
- [Repository hygiene and commits](#repository-hygiene-and-commits)

## Documentation

The README and model, adapter, weight, and performance guides are for users.
Explain technical details through their effects on performance, support,
reliability, correctness, usability, or reproducibility. Preserve precise public
API contracts, including shapes, valid configuration, and lifetime requirements.
Do not turn those guides into implementation status reports or instructions for
extending private APIs.

Put contributor rules here. STRUCTURE may contain implementation notes, package
responsibilities, and a code map. Keep design history and measurement receipts
separate from the current rules; this document describes the conventions to follow.

Give shared behavior one main explanation and link to it from the model guides.
Keep model-specific limits and runnable examples next to their workflows. Do not
copy a diagnostic script into several guides. Update affected docs when behavior
changes; do not preserve an obsolete default as a competing recommendation.

Use US English and plain ASCII in source, comments, docstrings, docs, and commit
messages. Use Unicode only when required by actual data or an upstream identifier.
Public functions and classes need a concise summary docstring, followed by details
when callers need them. Each module's docstring states what that module owns.

## Architecture and ownership

### Cohesive modules and explicit state

Aim for roughly 500 lines per file as a soft ceiling. Split by responsibility,
not alphabetical quotas or arbitrary line counts. A cohesive owner can exceed
that size; an unrelated helper does not belong there merely because space remains.

Prefer functions for math, pipeline composition, encoding, decoding, and state
transitions. Pass inputs explicitly and return results. Reserve classes for real
state: loaded weights, KV caches, open handles, runners, and active UI objects.
Avoid pipeline objects whose intermediate tensors accumulate on `self`.

### Model boundaries

Code under `kinomlx/models/<name>/` does not import a sibling model. Put genuinely
shared algorithms and contracts in neutral packages such as `modeling`, `samplers`,
`adapters`, `io`, and `media`. Share code for demonstrated reuse, not hypothetical
future models. Keep model-specific capabilities and policy with their model.

The root `kinomlx` namespace exports shared infrastructure. Import model requests,
runners, resources, products, and recipes from `kinomlx.models.<name>`; do not add
root aliases or model-specific re-exports of neutral helpers.

### Public model lifecycle

Every model follows the same ownership pattern:

1. A frozen typed request describes caller intent.
2. `prepare_resources()` resolves immutable source paths, capabilities, and policies
   without retaining live weights.
3. Component providers return bounded `ComponentLease` objects.
4. Stateless recipes consume requests, resources, and optional injected providers.
5. A model runner retains resources and providers across calls, not a live model bundle.
   Its `run(recipe, request)` boundary owns dispatch and operational error translation;
   named convenience methods expose meaningful recipes.
6. Model-owned products cross typed output/sink boundaries. Reserve deliverables
   before expensive work and publish related artifacts transactionally.
7. The model's `arguments.py` and `command.py` own its complete inference grammar and
   execution. The root CLI discovers and dispatches; it does not implement one model.

When adding a model, start with `types.py`, `resources.py`, `components.py`, a recipe,
`runner.py`, and command/output modules. Add transformers, sigmas, text encoders,
VAEs, conditioning, and converters only when that model uses them. A still-image
network must not inherit artificial diffusion or video concepts.

## Python style, imports, and types

KinoMLX requires Python 3.14 or newer and uses native deferred annotations.
Do not add `from __future__ import annotations`, `typing.TYPE_CHECKING`, or
`typing_extensions.TYPE_CHECKING`. Annotation dependencies are ordinary imports:
there is one import graph for execution and type checking.

Put imports at module scope, including MLX, native media, tokenizers, and diagnostics.
Function-local imports are permitted in tests, not package code. Resolve cycles by
moving a shared record, protocol, or alias to a neutral owner; do not conceal them
with quoted annotations, lazy exports, or type-checking branches. Package exports
use direct imports and explicit `__all__` lists. Importing the package or building
CLI help must not load weights, compile kernels, or start inference.

Use absolute imports across packages and relative imports within a package. Let
Ruff organize standard-library, third-party, and project imports. Ruff `TID251`
enforces the annotation restrictions; `PLC0415` enforces module-scope imports.

Use modern types: `X | None`, `list[X]`, and `dict[K, V]`. Public functions and
cross-module interfaces need complete annotations. Keep internal annotations useful
and satisfy the configured strict mypy gate; do not weaken types to bypass it.
Use `mx.array` for tensors and document relevant shapes and dtypes without inventing
unsupported generic tensor annotations.

| Element | Naming convention |
| --- | --- |
| Packages/modules | Lowercase; snake case when readability needs it |
| Model packages | Short, generation-stable names such as `ltx2`, without patch versions |
| Runners | `<Model>Runner` |
| Requests/configuration | Focused `<Recipe>Request` and `<Domain>Config` records; no universal pipeline config |
| Functions | Action-oriented snake case, such as `encode_text` or `denoise_step` |
| Private helpers | Leading underscore |
| Constants | `UPPER_SNAKE`, grouped under the module or record that owns them |

## Configuration and command boundaries

The frozen public configuration registry is the source of truth for model fields,
shared tables, CLI destinations/help, TOML parsing and serialization, templates,
environment discovery, and implications. Add a field through its owning schema;
do not manually mirror it across each input and output surface.

The shared model catalog lives in `kinomlx.model_registry`; saved configuration and CLI
consumers use the same frozen registry. Every `ConfigFieldSpec` explicitly declares
`persistence="intent"` for canonical input or `persistence="source"` for a spelling
that source resolution must consume. Saved readers, writers, and intent serialization
validate registered field shapes and reject unresolved source inputs. They do not silently
filter those inputs or defer malformed values until replay. Coupled final policy still runs
after explicit replay overrides have been applied.

Every registered model table owns its CLI bindings through `ConfigFieldSpec.cli`.
Tables set `cli_generated=True`; the model registry rejects a table that opts out,
and registration rejects missing field bindings. `add_config_arguments()` consumes
the complete contribution for all four models. A new ordinary field needs its typed
record field and bound schema metadata, without a parser-list edit. Primitive CLI
types derive from the record annotation; complex syntax and virtual fields declare
an explicit transport codec.

Declare aliases, repetition, optional values, and boolean actions in `CliBinding`.
Paired booleans can own distinct negative spellings and negative help, such as
`--reference-prompt-padding` and `--compact-prompt`. Settings keep their existing
field metadata and codecs; table construction compiles those facts into the same
binding contract. The standalone Settings helpers use that contract too. Do not add
a separate Settings parser or copy setting aliases into a second field inventory.

CLI binding coverage is complete for registered tables. Shared assembly, record
overrides, executable constraints, and restart policy ownership each have their own
declarations; CLI generation alone does not establish those global guarantees.

All model assemblers use `SetOverrides` for ordered `--set` operations. Source tables
merge recursively, including ordinary group-table assignments in `--set`. Scalar and sequence
assignments replace their targets; group assignments preserve unrelated members.
Normalize surviving overrides together as one layer, then apply them through the shared
replacement-aware merge. Do not collapse overrides into a dictionary and recursively merge
that dictionary into inherited state. The stream retains argument order and surviving explicit
leaf origins; those origins do not yet flow through saved-intent serialization.

Configuration rules declare `LayerTransform` inputs, writes, phase, and ordering dependencies.
Bind reusable rules to `ConfigFieldSpec.layer_transforms` so a field contribution brings its
behavior to every model. Model construction combines these with model-level rules, checks
field references and dependencies, and rejects conflicting rule identities or unordered
read/write overlaps. `ModelConfigSpec.normalize_layer()` runs this checked sequence around
registered implications. A rule receives only its declared present inputs and returns a
checked field patch; `None` is a clear marker and `DELETE_FIELD` consumes an input without
leaving an assignment. Deletion is subject to the same declared-write checks and preserves
explicit empty parent tables. Seed alternatives, output path selection, and LTX duration
selection use this mechanism. Effective rules run after precedence merging through
`normalize_effective()` and also serve resolved serialization; LTX inactive tiling geometry
uses that path. `prepare_layer()` first executes declared source rules and then pure layer
normalization. Prompt-file reads declare `read_files` effects,
which registration permits only in the source phase. LTX source selection compiles from owned
field groups with checked membership. Station applicability is a separate per-field declaration
described below; layer origins flow through assembly but not into saved-intent serialization.

Source preparation validates registered paths, table shapes, primitive types, and declared
choices before file reads or precedence merging. `parse_overrides()` admits every supplied
assignment before later replacements can hide it. Cross-field normalization still evaluates
surviving layer intent. Record annotations own input types; nullable virtual inputs declare
`input_annotation`, and field metadata cannot override a record-owned type. Reuse the shared
type coercer for both sparse admission and final typed construction.

`collect_invocation_sources()` owns ENV/file/CLI/override admission for all models. Source selectors
identify saved state before preparation executes contextual indexed edits or source effects.
A `[restart]` table with no `run` selects a restart only when a layer above the environment wrote
one of its keys; an environment-only table is ignored with a debug line naming the variables, and
any higher layer still makes `[restart].run` required, with the message naming the surface, or the
environment variable, behind each present key.
`collect_invocation_layers()` is the fresh-invocation composition of collection and preparation.
`InvocationLayers.merge()` owns ENV < saved base < file < CLI < ordered overrides. A saved base is
required typed configuration state and must be supplied during preparation, never attached later.
Caller-supplied Settings records suppress only their corresponding ENV tables. Model code
still owns saved-manifest selection, station-specific adjustments, and typed runtime construction.
Named layers are source context; complete per-field provenance remains separate work.

Sparse configuration output retains explicit source presence, including values equal to defaults.
Source decoding transfers presence to canonical fields; pure implications do not manufacture
explicit input. Prepared layers carry this presence through precedence into each invocation and
the shared print/save handler. Full and minimal TOML contain the same executable intent. Full output
adds commented built-in reference defaults; `--only-non-defaults` omits those comments. The exporter
requires retained intent and cannot accept effective snapshots. Field-owned pure serialization rules
restore alternative spellings, and a clear that TOML cannot reproduce fails before file creation.
Invocation records and output operations require metadata; there is no missing-metadata default.
Run schema version 2 stores required `kinomlx.configuration.v1` canonical values, explicit paths
and replacement markers separately from `effective_config` observations. Restart consumes canonical
state through
the current schema. Missing or unsupported formats fail without old-snapshot recovery.

Environment reads belong to the shared mechanism in `kinomlx.settings`, not scattered
`os.environ` calls. Bare settings construction uses deterministic field defaults;
environment resolution happens explicitly at invocation boundaries. Settings are
frozen and presence-tracked. Klein requests separate constructor-local seed bounds and choices
from final field-owned prompt, seed-selection and saved-conditioning rules. Call the request
validator on surviving effective inputs; generation recipes do so before consuming sources.
Tracked constructors and structured sources share coercion in
`config_values.py`; supplied primitive types are checked before post-init, nested records are built
through the same path, and type aliases retain their declared sequence behavior. Record state lives
below the coercer to avoid registry/Settings import cycles. Use `replace_config()` for supplied edits
and `derive_config()` for
resolved observations that retain the existing supplied-field set. `with_overrides()` uses tracked
copying and filters absent optional overrides. Ordinary runtime records can use `dataclasses.replace()`;
default-filled constructor arguments must not be used to reconstruct configuration presence.

Final record validation descends through nested dataclasses and record arrays, reporting concrete
member paths and indexes. LTX VAE tiling uses this boundary for overlap/size policy after merging;
its grid and nonnegative-value constraints remain local admission checks. Runtime tiling conversion
validates the same owned policy before constructing decoder plans.

Coupled Settings checks run through `EnvironmentSettings.validate()` after composition. A field
owns each pure `RecordRule`, including its name and complete read set. Registration rejects unknown
reads and conflicting rule identities; inspection exposes the same declarations. Primitive admission
must not run these checks before later layers can explicitly resolve a cross-field conflict. Final assembly
and resource boundaries run the validator even when a Settings record currently declares no rules,
so adding a field-owned rule cannot bypass an existing entry point.

Rules that distinguish an explicit default from an omitted input declare `presence_reads`
as a subset of their value read set. The callback receives immutable `RuleInputs` with only
those presence observations; missing tracking is an error, never inferred from value equality.
Shared adapter schedule policy uses this mechanism, so nested validation and direct adapter
selection enforce the same supplied-strength conflict.

Capability rules declare `phase="capability"` and their required `context_reads`.
Call `validate_capabilities(record, observations)` after preparing resource facts; it first
runs final input rules, then validates the declared observations recursively. Missing facts
fail explicitly. Callbacks receive immutable copies of only their declared context and cannot
perform source effects. Configuration inspection exposes both phases under `validation_rules`.
LTX uses this boundary for decoder kind, recipe, duration, keyframe, audio and conditioning
capabilities; input-only HDR relationships and keyframe positions remain final semantic rules.

Prompt-row grammar and explicit sigma-schedule admission have shared owners across image
requests and LTX stage schedules. Bind local constraints on every corresponding input field;
recipe and geometry helpers reuse those constraints when they accept resolved scalar values.

Restart selectors and concrete restart artifacts are tracked input records. Their fields own
choice inventories and final artifact relationships. Krea/Klein share still-image restart
field owners; LTX owns its stage-specific rules. Validate restart records before reading a run
manifest or consuming a selected artifact. Resolve an omitted latent stage when preparing the
restart, without inserting that derived default into the source record.

Registered model fields derive their environment names from the model ID and an owned
environment key, which defaults to the field name and is declared explicitly whenever the
public spelling differs. Parser storage names are not part of that public spelling, so
renaming a CLI destination cannot rename a variable. Registration rejects a field whose
name would claim another field's variable. Shared infrastructure owns shared bindings and
fallbacks. In particular, model output resolution uses the model-specific output directory,
then the shared output directory, then the record default. Do not add unprefixed
compatibility aliases or repeat derived environment names in model metadata.

Use the shared type parser for CLI/environment values. Boolean parsing accepts
explicit true/false spellings and rejects unknown words. Collections support TOML
arrays and the existing scalar-list syntax. Preserve literal text and meaningful
empty strings. CLI values do not undergo environment-template substitution;
explicit environment source templates are a separate mechanism.

Keep flag names and settings aligned through the registry and argparse bridge.
Provide explicit false controls for booleans where needed. Negative-only actions
use negative help text; paired boolean actions share the positive description.
Apply overrides as one typed operation rather than per-field mutation branches.

Semantic tags must have consumers. Model-specific groups belong to their model;
shared groups belong in the common enum only when shared behavior uses them.
Every output save field covered by the save-all implication must be included or
explicitly excluded, and an artifact toggle that follows save-all at runtime while it
stays unset is declared as inherited so the generated help states it. Reject
contradictory, ineffective, and unsupported combinations early rather than silently
ignoring them.

Write clear help text, but test structural and semantic contracts rather than a
required editorial vocabulary. CLI aliases do not imply Python or TOML aliases.
Public model APIs remain the execution surface used by the CLI.

### Declaring restart and public API ownership

Every contribution states its restart applicability. `ModelConfigSpec.restart_stations`
names the model's saved stations, or is empty with a `no_restart_reason` for a model that
has no saved restart operation. Each registered table covers exactly those stations, and
each non-virtual field declares one `StationPolicy` per station with a reason that
`kinomlx config inspect` prints. A policy is `mutable` when replay may use a new value,
`locked` when an earlier station already consumed it, or `irrelevant` when the station
cannot use it at all; an irrelevant policy also declares whether an explicitly supplied
value is rejected or ignored. Reuse the shared `settings_table()` and
`image_output_table()` builders, which require the model's stations, rather than
restating infrastructure policy per model.

Adding an ordinary field therefore needs its typed record member, its `ConfigFieldSpec`
with persistence, surfaces and station policies, and nothing else: no parser list, no
consumer inventory, no second policy table. Registration rejects a missing binding,
missing persistence, missing or duplicated station coverage, a policy without a reason,
an environment-name collision and a reused record member whose meaning was changed.

Public entry points declare parameter ownership with `APIBoundary`. A configuration
parameter refers to its registered table or to a projection of that table's record;
runtime ports such as component providers, reporters, callbacks, prepared resources and
station handoffs are classified as runtime-only, so they never gain an environment or
TOML representation. Validation compares declarations against the real callable
signature and annotations, requires projections to track constructor presence, and
therefore fails when a new parameter appears without an owner.

Every model's `run` hosts any recipe its ports fit, so its request is an unbounded
type parameter and is declared runtime on all four models; the concrete entry points
(`generate`, `restart`, `expand`) keep the typed configuration link to their
registered table. Registration refuses a runtime declaration whose annotation names a
registered table record, so a real configuration parameter cannot be relabeled as a
nonserializable port while an unbounded type parameter stays one.

## Weights and adapters

### Source loading and lifetime

Load original sources and prepare dtype, layout, decoding, and quantization lazily.
Do not reintroduce generated model-weight caches. Resource inventories describe
sources and consumed tensor keys; providers bind phase-owned components. Release
text encoders, transformers, and decoders at their phase boundaries. Selecting a
few components from a monolithic source must not evaluate unrelated tensors.

Safetensors is the runtime checkpoint format. Validate headers and supported
shapes/storage dtypes before binding. Resolve already-downloaded Hub snapshots
locally; ordinary generation does not download model weights. Source identities
and metadata digests are not claims that every tensor payload was hashed.

Checkpoint key mappings live as data under the owning model, separate from loader
control flow. LTX uses `weights/keys/`; adapter mappings use each model's
`adapters/keys.py`. Structure-driven VAE traversal helpers may remain beside the
iteration logic that determines their keys.

### Adapter ownership and fusion

`kinomlx/adapters/` owns shared parsing, format configuration, algebra, selection,
validation, and physical-layout fusion. Model adapter modules own actual target
mappings, exclusions, admission, and lifetime policy. Sharing the fuser does not
establish support for a format on another model.

Shared selections contain only common controls: path, total strength, exclusions, and
the shared per-step schedule or sigma window. LTX's independent stage totals remain
model-owned. Preserve each adapter occurrence as a composition term rather than
deduplicating paths. Use canonical parsed adapter types in reporting; filenames are
not format or architecture evidence.

Reconstruct equations in logical output-by-input coordinates, independently of
optimized physical layouts. Compute ordinary deltas against the same original base,
accumulate in a stable order in FP32, and cast to the target dtype. LTX's explicit
FP16 accumulation policy is a bounded exception; preserve its first-use checks.
For magnitude/rotation formats, strength scales the completed change from the base,
not a fractional internal normalization parameter. Saved full weights and trained
biases retain replacement semantics; compose saved bias contributions once.

Preserve format-specific alpha and scalar handling, including LTX's direct-strength
mapping, configured PEFT scaling, dense LoKr's unit-scale exception, and saved-zero
alpha semantics. The [adapter scaling reference](ADAPTERS.md#strength-alpha-and-effective-scale)
is the authoritative user explanation; do not duplicate the equations here.

Reject ambiguous target aliases, conflicting operations, and unknown semantic tensors
on recognized targets. Do not invent missing bases, mappings, or trainer state.
Record applied, excluded, unmapped, shared, and training-only tensors; shared bases
and training-only state must not inflate coverage. Unmapped updates warn even when
aggregate coverage is high. Low coverage warns rather than automatically failing.

Fusion leaves source checkpoints unchanged and is evaluated on first denoiser use.
A fresh transformer or changed stage profile needs fresh validation; unchanged retained
weights reuse their completed check. Zero-strength/excluded operations schedule no
weight scan. Disabling optional fused-weight checks does not disable format admission
or per-step latent checks. Preserve the distinction between evaluated finite outputs
and a claim that every intermediate value was finite.

### Converters

`kinomlx.weights` owns only generic, restricted tensor-only reconstruction, explicit
state/key selection, value/layout-preserving serialization, verification, and
transactional publication. Model-specific layout changes, folded constants, variant
identity, and loader verification belong to the model converter. Ambiguous checkpoint
branches require explicit selection; generic conversion must not guess one.

## Precision and numerical correctness

LTX, Krea, and Klein transformers default to BF16. FP16 is an explicit speed option;
both use two bytes per value. Keep the selected dtype coherent across projections,
feed-forward layers, attention, and activations. Unintended FP32 promotion changes
memory use and kernel selection. Diagnostic FP32 support remains model-specific.

Keep precision-sensitive calculations wider where the model requires it and cast
back deliberately. Preserve the source model's rounding, normalization, and frequency
contracts. An optimization that loads successfully is not evidence of numerical parity.

LTX-2.3 precision constraints:

- Honor `frequencies_precision=float64` when constructing the RoPE frequency grid;
  cast final cosine/sine tables to the hidden dtype.
- Stage-stable RoPE permits precomputation, but do not enable it by default without
  current end-to-end evidence that it improves performance.
- Keep sigmas, timestep bases, Euler/X0 updates, and noising scalars in FP32; cast
  the updated latent back to its payload dtype.
- Preserve the distilled checkpoint's `av_ca_timestep_scale_multiplier=1000`.
- Cast image/keyframe conditioning tokens to the latent dtype before concatenation.
- Reshape additive floating-point attention masks as well as boolean masks.
- Fused AdaLN must fall back when epsilon, dtype, or broadcast shape does not meet
  its kernel contract. Wider scale/shift work must cast back explicitly.
- A fused interleaved RoPE implementation reads input and cosine/sine values as FP32
  and casts only its final output to the activation dtype.
- Keep the BWE vocoder's FP32 computation and upstream output-dtype behavior.
  Returning FP32 instead is a separate quality experiment, not strict parity.

## MLX, native media, and memory

Keep model math, weight preparation, fusion, tokenization outputs, denoising, and
VAE execution in MLX-native arrays. NumPy is development-only for independent
oracles, synthetic fixtures, and diagnostics. Native media boundaries use Apple
buffers and explicit MLX buffer ingress, not convenience NumPy conversions.

Avoid hidden copies and synchronization:

- Do not wrap an already-typed MLX parameter in `mx.array()` to broaden its contract.
- Use MLX array creation, reductions, and shape operations instead of converting
  NumPy results or Python lists back into tensors.
- Do not convert lazy weight mappings to `dict`, `list`, or `tuple` without a reason.
- Evaluate the intended arrays explicitly, such as `mx.eval(*weights.values())`.
- Keep optional activation statistics behind diagnostics; scans materialize work.

Scalar constants, tokenizer IDs, Python-authored index tables, and byte/buffer
boundaries remain valid uses of `mx.array()`. Matching a reference requires checking
its actual arithmetic: similarly named operations, including `linspace`, need not
produce bitwise-identical results across libraries.

Use native Apple media frameworks where available: ImageIO/CoreImage for stills,
AVFoundation for audio and file writing, and VideoToolbox for video processing.
Keep frame/signal contracts model-neutral and native failures typed. Verify native
API selectors against authoritative interfaces rather than guessing their spelling.

Memory policy is centralized at invocation boundaries. Resource inspection has no
MLX-limit side effects; runners and public recipes apply both limits before model
work. Standalone component hosts explicitly apply their policy. Do not scatter
model-local limit setters or introduce hidden cache floors. The allocator default
is 1 GiB; wiring follows the effective system recommendation. Preserve the distinct
zero semantics documented in [Performance](PERFORMANCE.md).

The LTX full-residency target is 64 GB, with streaming for supported 32 GB workflows.
Do not generalize those targets to every model or resolution. Report insufficient
memory clearly and suggest only supported alternatives. MLX evaluation alone does
not prove that active weights remain resident rather than compressed.

## Reporting and profiling

Runtime code uses stdlib logging and `Reporter`; it does not import Rich, call
`print`, own stdout, or branch on CLI presentation flags. Hosts may inject null,
recording, or custom reporters. `kinomlx.ui` owns Rich presentation on stderr.

Keep machine stdout isolated for structured results or deterministic configuration
text. Quiet mode suppresses human presentation while preserving phase accounting.
Configuration printing and saving use the same serializer.

Fit run-header label/value pairs to terminal width without splitting or shortening
individual items. Oversized items and prompts get their own lines. Avoid duplicate
facts, routine fingerprints, trailing row padding, and repeated sampling/output
announcements. Tests should inspect meaningful content, not exact indentation.

Profiling is opt-in. Compile the native signpost shim only when enabled, using the
system compiler and a source/architecture-keyed application cache. Do not commit
architecture-specific binaries. Failure to build the optional shim disables
signposts without aborting ordinary generation.

## Dependencies and provenance

Declare runtime dependencies in `project.dependencies` and development dependencies
in the dev extra. Prefer minimum supported versions without speculative upper bounds.
Do not add dependencies before a concrete use exists or change tool pins incidentally.
Use a project virtual environment and `uv pip` for dependency management.

Keep inference independent of Torch, Diffusers, Transformers, PEFT, LyCORIS, and
other producer frameworks. Read upstream implementations as specifications where
appropriate; adaptation requires compatible licensing, exact provenance, and retained
notices. Keep upstream comparison environments separate from the runtime dependency
set so reference frameworks do not become accidental inference requirements.

## Tests and validation

Run tests from the repository root with the project virtual environment active.
MLX tests require an Apple Silicon environment with access to Metal.

Tests mirror package responsibilities under `tests/`. Add tests for meaningful
regressions, public contracts, and independently derived numerical behavior, not
coverage totals, nonexistent features, or a restatement of the implementation.
Use injected providers to test lifecycle and failure behavior without real weights.

The suite runs in parallel by default, so every new test must pass under arbitrary
worker interleaving: no shared module state, no dependence on a neighboring test,
a named `logger=` on every log-capture level, and written files under the test's own
temporary directory. Only a `heavy` test may assume a single process, and those are
forced serial. Run `-n 0` for coverage measurement and deliberate order experiments;
do not reach for a grouping option that preserves file order to keep an
order-dependent test passing, and do not write a test that needs one.

Tests carry three layers. A `heavy` test loads a real checkpoint or runs long GPU
work: it is deselected by default, selected with `--run-heavy`, always runs in one
process, and refuses to start if it reaches a parallel worker by any other route.
`slow` sub-labels the heavy tests that take minutes, so `--run-heavy -m 'heavy and
not slow'` runs the quick checkpoint checks. Every other test runs in parallel on
every invocation, including the cheap real-asset tests: a header read, tokenizer
parity, or one small component load resolves its input the way the product does and
skips with the reason when the cache does not hold it. Registered markers are the
only markers a test may use (`--strict-markers`).

Run the repository's configured gates for the changed surface. From the repository
root with its virtual environment active:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
rumdl check .
git diff --check
```

Ruff lint and formatting are distinct gates. Use the same selected Ruff executable
throughout; format only changed files unless a broader normalization is intentional.
Use configured mypy targets rather than indiscriminate workspace traversal.

`python -m vulture` is an occasional sweep rather than a gate; `[tool.vulture]`
already carries its paths and the names that dynamic callers resolve. Read its
unused function, method, property and class findings first. A field of a declared
record surfaces as an unused attribute because generic intent capture and
serialization reach it by name, so those lines are expected. Ruff's `ARG` rules
cover the related live case: a parameter the body never reads is either an
interface obligation, which says so with a `noqa` and a reason, or a configuration
input the code silently ignores, which is a defect to fix rather than to suppress.

For documentation-only changes, run Markdown lint and relevant link/command/API
contract checks. Do not run the entire model suite solely because prose changed.
Changed Python documentation tests still need their Ruff checks. Runtime, packaging,
or command-behavior changes require the applicable additional tests. For package
validation, verify that imports actually resolve from the built/installed artifact.

## Performance measurements

Keep exploratory benches and native probes outside the installable product.
Reuse existing characterization before starting another benchmark. Trace production
call sites to verify shapes, dtypes, broadcast patterns, masks, layouts, and lifecycle;
plausible-looking synthetic dimensions are not enough. Profile the time budget first.

Apply both allocator and wired-memory limits before tensor allocation or model work,
outside the timed region. Use the same default memory policy as production and record
both effective limits and the system wired setting. Hold memory policy constant
across comparisons unless that policy is the variable being measured.

Compare alternatives with the same code revision, interpreter, MLX build, OS/driver,
and hardware. Run adjacent repeated controls with thermal and background-load
conditions recorded. Separate preparation, first-use materialization/fusion, steady
execution, and output work. Measure production shapes and end-to-end impact; pair
speed claims with numerical/image parity, resource use, and a usable fallback.

Record the code revision, environment, inputs, and measurement procedure with
benchmark results so others can reproduce them. A faster operation matters only
in proportion to its share of the actual workload.

## Repository hygiene and commits

Use portable paths, documented environment variables, and neutral examples. Keep
personal information, machine-specific paths, and private workflow references out
of source and documentation. Retain required third-party attribution.

Keep commits focused on a coherent change. The commit subject is the first line:
use an imperative title, such as "Clarify adapter scaling", and aim for roughly
70 characters or fewer. This is a readability guideline, not a hard limit. After
a blank line, use the body to explain the problem, the changes, and relevant rationale
in as much detail as needed. A broad change can have a short summary title and a
detailed body.

When changing a project convention, update this document along with the affected
code and explain the reason in the change description.

### Prepared Klein recipe policy

Klein request fields own negative-row cardinality and source-geometry relationships as
final input rules. These checks do not read source images. Recipe capability rules require
an explicit variant observation and validate external CFG, negative conditioning and adapter
schedule lengths against the resolved recipe. Direct sampling recipes use the same profile
default resolver and CFG policy; their constructors admit local values and their `resolved()`
operation validates final relationships. An explicit sigma schedule owns its execution step
count without overwriting the separately retained input step count.

### Station output availability

Artifact fields can bind an `OutputPolicy` declaring the stations that produce them.
Registration checks the boolean owner and station identities. Image restart readers derive
unavailable parent output defaults from this policy. Shared invocation preparation filters
unavailable implications in every current source layer while retaining explicit true and
false requests and their provenance. An explicit unavailable request fails validation even
when save-all is enabled. Both image output planners consume the same declarations.

Station output availability describes artifact production; it does not replace the separate
mutable/locked comparison policy for request inputs. A restart station re-emits what it
consumed and still holds, so Krea and Klein decode both publish final latents and both
denoise stations publish text conditioning again. Krea restricts text conditioning to its
text recipe, so an edit run cannot save a text-only restart.

### Invocation utility controls

`kinomlx.invocation_controls` owns bootstrap and configuration utility inputs: their names,
typed CLI bindings, defaults, help, phase, dependencies and surface exclusions. Bootstrap
and complete model parsers consume the same declarations. Parser validation checks these
controls as well as configuration fields; they are no longer an unchecked destination list.
The model selector links to its canonical field and shares its binding with each model schema.

Configuration inspection includes invocation controls and their supported/excluded surfaces.
Config-file selection, ordered override input and print/save modes are command utilities;
they do not become fake TOML fields or saved generation values. Utility dependencies, such
as the output mode required by `--only-non-defaults`, are checked by the shared utility boundary.

### Required configuration surface policy

Each `ConfigFieldSpec` explicitly selects a `SurfacePolicy`. The shared policies cover normal
typed inputs, source spellings, model selection and structured-only contributions. A policy
reports every transport as supported or deliberately unavailable with a reason. A field cannot
claim ordinary CLI support without a binding, or retain a binding while excluding CLI. Deliberate
ENV exclusions remove the binding; supplying its unregistered variable still fails strict ENV
admission. Settings records retain their owned ENV bindings; they cannot independently exclude
that same surface in schema metadata. Collection members must support the transports admitted
by their parent collection. Virtual source spellings explicitly exclude typed-record API arguments.

Ordinary configuration fields participate in TOML and ordered overrides and expose their typed
record members through the API. Runtime objects and command utilities belong to their distinct
boundary declarations. Configuration inspection reports field and nested-member policies and
invocation control exclusions directly from these declarations.

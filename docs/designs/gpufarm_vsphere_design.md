# gpufarm: vSphere-for-GPUs Capability Tier — Design

Date: 2026-06-12
Status: draft, pending operator review of phasing + open questions

**Scope:** evolve gpufarm from a priority-ordered first-fit dispatcher with ad-hoc healing into a declarative, scored, reconciling farm manager. Constraints honored: single-host coordinator + SQLite (`store.py:Store`), YAML-in-git manifests, full backwards compatibility for existing `jobs.yaml` entries, heterogeneous fleet (2x Blackwell 96GB, 4x GB10 128GB, RTX 5090 32GB, Strix Halo, overlapping `cluster-4node`).

## 1. Gap analysis

| # | Capability | Exists today | Missing | Effort |
|---|---|---|---|---|
| 1 | Pin to resource | `Job.requires_resource` (models.py:703), enforced in `_matches()` (scheduler.py:219) | Soft pinning (prefer-but-fallback); pin-to-host (vs. resource id) | S |
| 2 | Affinity / anti-affinity | Nothing first-class. Approximations: `Resource.can_run`/`avoid` (models.py:417-419), comments like "operator-mutex" for `blackwell-host` vs `blackwell-gpu0/1` and `cluster-4node` vs `gb10-*` (resources.yaml:60-67, 179-185) -- pure honor system | Job-to-job anti-affinity, job-to-endpoint affinity, and critically a **hardware-overlap mutex** so overlapping resource views can't double-book a physical GPU | M |
| 3 | Priority classes + preemption | `Job.priority` 0-9 (models.py:699), queue ordered `priority ASC, submitted_at ASC` (store.py:458), tier preference split at `priority < 5` (scheduler.py:196). `preemptible` flag parsed (manifests.py:285) but **never read by any scheduler/coordinator code**; `RunStatus.KILLED_PREEMPT` (models.py:45) is never assigned | Named priority classes with consistent semantics; actual compute-job preemption (kill + requeue); preemption cost in match decisions | M |
| 4 | Requirements matching | `vram_gb` gate via `estimated_vram_gb` + `vram_free_by_resource` (scheduler.py:232-238, built in coordinator.py:2213-2241); class via `resource_class` list (scheduler.py:225); python env only as `python_bin` *override*, not a matchable constraint -- the #162/#163 venv-mismatch class of failure is documented in resources.yaml:78-83 | Declarative job-side `requires:` block (vram, labels/capabilities, python env) matched against resource-side `provides:`/`labels:`; today's "exclude job from `can_run` per resource" workaround inverts ownership | M |
| 5 | Reservations + queue adapting | Full reservation FSM (`ReservationState`, models.py:194), `match_reservations()` (scheduler.py:245), reservation-held masking via `resources_with_reservations` (scheduler.py:134), external-lease preemption + restore (`preempts_external_lease` models.py:809, `_restore_preempted_lease` coordinator.py:3269) | Queue *lookahead*: a QUEUED reservation pinned to a resource should block long compute jobs from starting there while allowing short backfill; today a queued reservation only takes effect once dispatched | M |
| 6 | Inference unloadable/drain rules | `ExternalLease.state` HELD/RELEASED/AUTO/PREEMPTED (models.py:54-96), `drain_seconds_on_release` (models.py:676), `_drain_if_due` (coordinator.py:2138), restore guarantee via `set_external_lease_force_state` (store.py:1207) -- but "may this be evicted" lives on the **job side** (`preempts_external_lease`), not on the model/lease | Lease-side `eviction:` policy (`unloadable`, drain seconds, restore guarantee, who may evict). Today any job manifest can set `preempts_external_lease: true` and the lease can't refuse | S-M |
| 7 | Best-GPU fitting | First-fit by `(class_rank, +/-priority_tier, id)` (scheduler.py:197-203); VRAM gate is binary filter, not a score | Scored-fit: VRAM headroom best-fit, affinity bonuses, preemption cost; operator-tunable weights | M |
| 8 | Self-managing reconciliation | Pieces exist: startup `_reconcile_running` (coordinator.py:1617), `RecoveryManager.observe` ladder (recovery.py:75), supervisor restart budgets (models.py:910), lease probe `resolve_state` (external_lease_probe.py:239), endpoint drift probe (endpoint_health_probe.py:192), rogue/external detection `_poll_external_gpu_use` (coordinator.py:956) | A coherent observe-plan-act tick contract; rogue + drift signals feed back into *scheduling and repair*, not just events (`ENDPOINT_DRIFT` and `ROGUE_WORKLOAD` are emit-only today) | L |

## 2. Schema design

All additions extend existing files and are optional -- every current entry parses unchanged (the `_LenientModel` pydantic shadows in manifests.py already ignore unknown keys, so rollout can even be staged).

### 2.1 `resources.yaml` -- scheduler block grows priority classes + scoring policy; resources grow `labels`, `provides`, `shares_hardware_with`

```yaml
scheduler:
  default_resource_class_priority: [blackwell, gb10, rtx5090, strix_halo]

  # NEW: named priority classes. Jobs reference by name; numeric `priority`
  # keeps working (mapped to the class whose band contains it).
  priority_classes:
    - name: training          # cluster FSDP rounds
      priority: 0
      preempts: [backfill]    # may evict running runs of these classes
      grace_seconds: 120      # SIGTERM -> SIGKILL window on eviction
    - name: interactive       # serve jobs, operator one-offs
      priority: 2
      preempts: [backfill]
      grace_seconds: 60
    - name: evals             # per-ckpt evals (today priority 4-6)
      priority: 5
    - name: backfill          # gen_samples etc; evictable by default
      priority: 7
      preemptible: true       # class-level default; job-level flag overrides

  # NEW: scoring policy (weights for the scored-fit matcher, section 3).
  scoring:
    vram_fit_weight: 1.0        # best-fit: prefer smallest sufficient headroom
    class_rank_weight: 10.0     # keep resource_class order dominant
    tier_weight: 2.0
    affinity_weight: 3.0
    preemption_cost_weight: 20.0   # evicting anything must be clearly worth it

resources:
  - id: blackwell-gpu0
    # ... everything existing unchanged ...
    labels:                       # NEW: free-form capability labels
      python_env: vllm-turboquant
      cuda: "12.1"
    shares_hardware_with: [blackwell-host]   # NEW: physical-overlap mutex
  - id: blackwell-host
    shares_hardware_with: [blackwell-gpu0, blackwell-gpu1]
  - id: gb10-spark
    shares_hardware_with: [cluster-4node]
  # cluster-4node lists all four gb10-* views; loader validates symmetry.
```

`shares_hardware_with` formalizes the "operator-mutex" comments at resources.yaml:60-67 and 179-185: a resource is unmatchable while any overlap sibling is BUSY/LEASED or reservation-held. This is the single highest-correctness-value schema addition -- today nothing stops a `consolidate_single_host` dispatch while `blackwell-gpu1` runs an eval.

### 2.2 `jobs.yaml` -- declarative requirements + affinity + priority class

```yaml
  - id: reasoning_eval
    script: training/reasoning_eval.py
    resource_class: [blackwell]          # unchanged, still honored
    priority_class: evals                # NEW (numeric `priority` still accepted)
    requires:                            # NEW: declarative job-side requirements
      vram_gb: 24                        # supersedes estimated_vram_gb (alias kept)
      labels:
        python_env: vllm-turboquant      # matched against resource.labels --
                                         # kills the #162/#163 rc=127 class of bug
    affinity:                            # NEW
      prefer_resources: [blackwell-gpu1] # soft pin: bonus, not a filter
      anti_affinity:
        jobs: [sft_lora]                 # never co-host with a running sft_lora
        scope: host                      # resource | host
        mode: hard                       # hard = filter; soft = score penalty
      near_endpoint: gpt-oss-120b        # prefer same host as this SERVING model
```

Back-compat rules in `Manifests.load` (manifests.py:639): `requires.vram_gb` defaults from `estimated_vram_gb`; absent `priority_class` derives from numeric `priority`; absent `affinity` means no change in behavior.

### 2.3 `models.yaml` -- lease-side eviction policy

```yaml
  - id: gpt-oss-120b
    family: vllm
    vram_gb_min: 78
    vram_gb_max: 96
    default_resource_class: blackwell
    eviction:                       # NEW: the model/lease decides, not the job
      unloadable: true              # false = nothing may preempt this lease, ever
      drain_seconds: 60             # wait for in-flight requests before stop
      min_evictor_class: interactive  # only >= this priority class may evict
      restore: guaranteed           # coordinator must restart on evictor release
                                    # (today's _restore_preempted_lease behavior,
                                    #  now contract rather than side effect)
    launch_by_resource:
      blackwell-gpu0: { ... unchanged ... }
```

`Job.preempts_external_lease` (models.py:809) remains as the *request*; the lease's `eviction.unloadable` + `min_evictor_class` become the *grant*. A `unloadable: false` model (e.g. the production coder endpoint during a campaign) refuses even `preempts_external_lease: true` jobs -- the scheduler logs the refusal as an event instead of silently matching.

## 3. Scheduler: first-fit to scored-fit

`match()` (scheduler.py:105) stays a pure function and keeps its exact signature philosophy -- all I/O-derived state arrives as arguments. The change is internal: replace the sort at scheduler.py:197-203 with a scoring pass.

```python
@dataclass(frozen=True, slots=True)
class MatchContext:
    resources_with_reservations: dict[str, int]
    queued_reservations: list[Reservation]       # NEW: lookahead input
    vram_free_by_resource: dict[str, float]
    running_jobs_by_resource: dict[str, str]     # NEW: for anti-affinity + overlap
    lease_eviction_policy: dict[str, EvictionPolicy]  # from models.yaml
    scoring: ScoringPolicy                       # weights from resources.yaml

def score(job: Job, run: Run, resource: Resource, ctx: MatchContext) -> float | None:
    """None = hard-infeasible. Higher = better. Pure, deterministic."""
    if not _matches(job, resource, ctx):          # existing gates + labels + overlap
        return None
    s = 0.0
    # 1. class preference (dominant, preserves v0.7 semantics)
    s -= ctx.scoring.class_rank_weight * job.resource_class.index(resource.class_)
    # 2. VRAM headroom best-fit: smallest-sufficient wins -> a 24GB eval
    #    lands on the 32GB 5090, not a free 96GB Blackwell
    free = ctx.vram_free_by_resource.get(resource.id)
    if free is not None and job.requires.vram_gb:
        s -= ctx.scoring.vram_fit_weight * (free - job.requires.vram_gb)
    # 3. tier alignment (existing rule, as score not sort)
    tier = resource.priority_tier
    s += ctx.scoring.tier_weight * (-tier if run.priority < 5 else tier)
    # 4. affinity bonus / soft anti-affinity penalty
    s += ctx.scoring.affinity_weight * _affinity_delta(job, resource, ctx)
    # 5. preemption cost: claiming a HELD lease or evicting a preemptible
    #    run subtracts cost scaled by what gets killed
    s -= ctx.scoring.preemption_cost_weight * _preemption_cost(job, resource, ctx)
    return s
```

The greedy outer loop is unchanged: queued runs in priority order, each takes its argmax-score resource, chosen resource removed from the pool -- so existing first-fit tests describe a special case (weights collapse to the old sort key, which is the Phase 1 compatibility test).

**Reservation queue-adaptation hook:** the new `ctx.queued_reservations` input. For each QUEUED reservation pinned to resource R (the only mode today, scheduler.py:257), `score()` applies a *time-aware penalty* to compute candidates on R: if `job.estimated_minutes` exceeds a configurable backfill window (default: time until the operator expects the reservation to start, or a flat `reservation_backfill_max_minutes: 30`), R is infeasible for that job; short jobs still backfill. This is exactly the "queue adapts around reservations" requirement, and it lives in the pure function -- the coordinator merely builds the list from `Store.list_reservations(states=[QUEUED])`.

**Preemption output:** `match()` grows a second return channel -- `list[PreemptionPlan(victim_run_id | lease_resource_id, evictor_run_id, grace_seconds)]`. The coordinator executes plans before dispatching the evictor: kill via the existing `_kill_run_pid` (coordinator.py:4101) path, mark `RunStatus.KILLED_PREEMPT` (finally used), requeue the victim if it has retries, and for leases reuse the existing stop/PREEMPTED/restore machinery (coordinator.py:3162-3285).

## 4. Reconciliation loop -- what "self-managing" means per tick

`_tick` (coordinator.py:794) already has the right ordering instinct (observe before dispatch). The redesign names the contract explicitly -- three stages, same single async loop, no new processes:

**Observe** (exists, keep): `_poll_health` -> `_poll_external_lease_probes` (resolve_state, external_lease_probe.py:239) -> `_poll_external_gpu_use` (rogue + external-busy, coordinator.py:956) -> `EndpointHealthMonitor.poll_if_due` (endpoint_health_probe.py:192) -> `_reap_running` / `_reap_reservations`.

**Plan** (new, the gap): a pure `plan(desired, observed) -> list[RepairAction]` step between reap and dispatch. Desired state = manifests + queued runs + non-terminal reservations + supervisors. Observed = health snapshots, lease states, rogue details, endpoint probe results. Divergences and their repairs:

| Divergence | Today | Plan-stage repair |
|---|---|---|
| `ENDPOINT_DRIFT` (served_name mismatch, endpoint_health_probe.py:370) | event only | feed the reservation's `on_health_failure` pipeline (`_on_health_failure`, coordinator.py:2032): RESTART relaunches the declared model -- drift becomes a self-correcting condition, not a dashboard row |
| `ROGUE_WORKLOAD` on a BUSY resource | event only | subtract rogue `vram_mb` from `vram_free_by_resource` for scoring; if the co-located gpufarm run has `requires.vram_gb` no longer satisfiable, emit a HEALTH_ALERT escalation (kill-rogue is deliberately out of scope -- operator policy question Q3) |
| `external_busy` on IDLE resource | masks scheduling (works) | unchanged; additionally counted as occupancy in scoring so near-term planning sees it |
| Lease declared HELD but probe says inactive + VRAM below threshold | doctor surface only | with `eviction.restore: guaranteed`, plan a service restart (the restore obligation generalized from `_restore_preempted_lease`) |
| GPU fault | `RecoveryManager.observe` ladder (recovery.py:75) | unchanged mechanism, but ladder decisions become RepairActions in the same plan log, giving one auditable "what the farm decided this tick" stream |

**Act** (exists, keep): execute RepairActions under existing budgets (recovery window/backoff, supervisor restart windows, reservation health-failure counters), then `_dispatch_queued` with the scored matcher. Preempt-vs-queue-vs-scale-down policy: preempt only when `score_gain > preemption_cost` *and* the priority class grants it; queue otherwise; scale-down (releasing an idle auto-extend reservation early) only via the existing TTL/idle machinery (`_handle_ttl_window`, coordinator.py:2078) -- no new kill paths.

Every plan is logged as a `RECONCILE` event before execution, and `CoordinatorConfig` gains `reconcile_mode: observe|repair` (mirroring `RecoveryMode.DRY_RUN`, models.py:384) so the operator can watch a week of "would-do" before enabling.

## 5. Phased delivery

**Phase 1 -- declarative requirements + scored-fit (highest leverage, smallest slice).**
Changes: `models.py` (add `JobRequirements`, `ScoringPolicy` dataclasses; `Resource.labels`); `manifests.py` (`_JobY.requires`, `_ResourceY.labels`, `_SchedulerConfigY.scoring`, defaulting from `estimated_vram_gb`/`priority`); `scheduler.py` (`score()` + `MatchContext`, replace sort); `coordinator.py:_dispatch_queued` (build context -- it already builds `vram_free_by_resource`).
Tests: pure unit tests asserting (a) default weights reproduce today's exact assignments on fixture fleets (golden-master against current `match()`), (b) best-fit lands a 24GB job on the 5090 when a Blackwell is also free, (c) label mismatch excludes the gb10 nodes for `inference_benchmark` (regression for #163).
Ships alone: zero manifest edits required; behavior identical until an operator adds `requires:`/`scoring:`.

**Phase 2 -- hardware-overlap mutex + affinity/anti-affinity.**
Changes: `manifests.py` (parse + validate `shares_hardware_with` symmetry); `scheduler.py` (`_matches` consults `running_jobs_by_resource` + overlap set; affinity scoring term); `coordinator.py` (populate running-by-resource from `store.list_runs`); resources.yaml gains the overlap declarations replacing the honor-system comments.
Tests: cluster-4node busy means all `gb10-*` views infeasible and vice versa; `blackwell-host` blocked while either GPU view runs; supervisor on cluster-4node also blocks (supervisors hold no lease -- derive occupancy from active supervisor runs, store.py:905).
Ships alone: converts documented operator footguns into enforced invariants.

**Phase 3 -- priority classes, real preemption, lease eviction policy.**
Changes: `models.py` (`PriorityClass`, `EvictionPolicy`); `manifests.py` (scheduler.priority_classes, model `eviction:`); `scheduler.py` (PreemptionPlan output; `_preemption_cost`); `coordinator.py` (execute plans: `_kill_run_pid` -> `KILLED_PREEMPT` -> requeue victim; route lease evictions through the existing PREEMPTED/restore path gated by `eviction.unloadable`); `store.py` (no schema change -- `KILLED_PREEMPT` already exists).
Tests: priority-0 train run evicts a backfill `gen_samples_multidepth` and the victim requeues; `unloadable: false` lease refuses a `preempts_external_lease: true` job with an event; preemption never fires when a plain-idle alternative scores adequately.

**Phase 4 -- reconciliation plan stage + probe feedback.**
Changes: new `gpufarm/reconcile.py` (pure `plan()` + `RepairAction`); `coordinator.py:_tick` inserts plan/act between reap and dispatch; `endpoint_health_probe.py` results routed into `_on_health_failure`; rogue VRAM subtraction into the Phase-1 context; `CoordinatorConfig.reconcile_mode`.
Tests: drift fixture -> RESTART plan; rogue-VRAM fixture shrinks scoring headroom; observe-mode produces identical store state to today (no-op guarantee).

## 6. Open questions for the operator

1. **Compute preemption semantics:** none of the eval scripts checkpoint. Is kill-and-requeue-from-zero acceptable for `preemptible: true` jobs (losing up to `estimated_minutes` of work), or should preemption be restricted to jobs under some duration threshold?
2. **Overlap mutex strictness:** should `shares_hardware_with` be scheduler-only (new submissions blocked) or also retroactive -- i.e., should the reconciler flag/act when it *finds* an overlap violation already running?
3. **Rogue workloads:** detection is now solid; is the desired ceiling forever "mask + alert," or do you want an opt-in `rogue_action: kill` per resource (with the same dry-run ladder discipline as recovery)?
4. **Bin-packing direction:** the sketch best-fits VRAM (consolidate small jobs onto small cards, keep Blackwells free). For the GB10s, do you instead want *spread* (thermal/risk distribution across the four nodes) when scores tie?
5. **`unloadable: false` vs. recovery ladder:** may a `RecoveryLevel.REBOOT` action still take down a host carrying an unloadable lease (availability vs. fault-repair priority), or does unloadable also veto host reboots?

## Critical files for implementation

- /Users/alex/git/gpufarm/gpufarm/scheduler.py -- `match()`/`_matches()` become `score()`-driven; all new matching semantics land here as pure functions
- /Users/alex/git/gpufarm/gpufarm/models.py -- new dataclasses: `JobRequirements`, `PriorityClass`, `EvictionPolicy`, `ScoringPolicy`, `Resource.labels`/`shares_hardware_with`
- /Users/alex/git/gpufarm/gpufarm/manifests.py -- pydantic shadows + back-compat defaulting (`_JobY`, `_ResourceY`, `_SchedulerConfigY`, `_ModelY`)
- /Users/alex/git/gpufarm/gpufarm/coordinator.py -- `_tick` plan/act insertion, `_dispatch_queued` context building, preemption-plan execution
- /Users/alex/git/OpenMythos/gpufarm/resources.yaml -- operator-facing rollout surface (scoring block, priority classes, overlap declarations)

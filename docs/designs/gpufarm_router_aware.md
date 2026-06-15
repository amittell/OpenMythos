# Router-Aware gpufarm: Unified Endpoint Catalog

Status: design / proposed
Date: 2026-06-14
Author: Alex Mittell
Task: #172 (Router-aware gpufarm: endpoint catalog CLI + web UI + client/server awareness)

## 1. Motivation

Operator request, verbatim:

> "route through mini-beast's :8080 router, every gpu host has a router that
> gpufarm should go through. it would be cool if gpufarm cli could return
> available endpoints and models to me, and show endpoints for easy
> click-to-copy in the gpufarm web ui too. perhaps we should go further and
> make the routers and gpufarm client/server aware kinda thing?"

Today an operator who wants to point a client at a model has to remember which
host runs it, that the router listens on `:8080`, and that the curl-ready base
URL is always `http://HOST:8080/v1` (the per-backend `127.0.0.1:80xx` URLs are
internal and not reachable off-host). gpufarm already knows the fleet topology
and already polls the routers for one narrow purpose (conflict discovery), but
it does not expose a unified catalog of "what can I call, and where."

This design adds that catalog: discovery, an aggregated in-memory view, a
`gpufarm endpoints` CLI, and a web-UI panel with click-to-copy. It then
phase-gates the deeper "client/server aware" idea (registration + fleet-wide
routing) behind the read-only first slice.

### Fleet reality (from the router survey)

- Only two of seven hosts run inference routers, both on `:8080`, both
  OpenAI-compatible, both with health at `GET /health` (NOT `/healthz` or
  `/v1/health`):
  - `kebab-rtx6000.lan` -- FastAPI/uvicorn, `kebab_rtx_router_fastapi.py`,
    2x Blackwell PRO 6000. `GET /v1/models` is DYNAMIC (probes upstream vLLM
    and reports actually-served ids).
  - `mini-beast.lan` -- custom `http.server` router (`router.py`, 237 KB),
    RTX 5090 + AMD Strix Halo 8060S behind ONE router. `GET /v1/models` is a
    STATIC hand-coded list of 18 OpenAI capability names; live loaded state
    comes from `GET /backends` and `GET /admin/models/status`.
- The 4-node GB10 training cluster (`kebab-spark`, `kebab-gx10`, `-2`, `-3`)
  runs NO routers and NO HTTP servers (only sshd). They are training-only.
- `mac-mini.lan` is the gpufarm COORDINATOR (dashboard `:8765`), not an
  inference router. It has no `/v1/models`, no `/catalog`. It is the natural
  place to HOST the unified catalog (it already polls the fleet), not to serve
  models.

Design consequence: gpufarm keeps a static registry of router endpoints, polls
each live for served names and loaded/unloaded state, and flattens everything
to client-ready rows. The catalog lives on the coordinator. GB10 and mac-mini
are recorded as `has_router=false`.

## 2. Gap analysis

| Capability | Where | Status |
|---|---|---|
| Per-resource `router: {url, capabilities}` block | `resources.yaml:212-214, 270-272, 326-328, 386-388, 466-468` | EXISTS |
| `RouterConfig` dataclass (url, capabilities, auth_env_var, timeout, health_path) | `transports/types.py:91-114` | EXISTS |
| `RouterCapability` enum (HEALTH, STATS, METRICS, BACKENDS, MODELS_LIST, ADMIN_MODELS_*) | `transports/types.py:18-43` | EXISTS (`MODELS_LIST`/`BACKENDS` defined but not declared on any resource yet) |
| `RouterClient` async HTTP wrapper (`has`, `get`, `post`, `health`, `model_status`) | `transports/router.py:79-269` | EXISTS |
| `Model.launch_by_resource` with per-resource `port`, `served_name` | `models.py` (`ModelLaunchSpec`), `models.yaml:32-127` | EXISTS |
| Endpoint base_url derivation `http://host:port` | `api.py:138-151` + `endpoint_health_probe.py:72-89` (`_reservation_endpoint`) | EXISTS (duplicated) |
| `EndpointHealthMonitor` (poll reservation endpoints, drift detection) | `endpoint_health_probe.py:143-373` | EXISTS |
| `_endpoint_health_cache` populated each tick | `coordinator.py:526`, filled at `coordinator.py:991` | EXISTS |
| Router poll for conflict discovery (`/admin/models/status`) | `coordinator.py:2271-2356` (`_refresh_router_conflicts`) | EXISTS (reads router, does NOT catalog served models) |
| CLI HTTP-online pattern (`_coordinator_request` + Rich table) | `cli.py:791-808` (helper), `811-871` (`status`), `1174-1186` (`queue`), `1255-1267` (`rogue`) | EXISTS |
| `GET /api/queue`, `/api/rogue`, `/api/reservation-health` API routes | `api.py:309, 924, 894` | EXISTS |
| Dashboard view-builder + HTMX partial + route pattern | `dashboard.py:949` (`_build_health_view`), `1272-1285` (`dashboard_health` route) | EXISTS |
| `data-action` delegated click handler + `toast()` | `static/app.js:363-388` (handler), `28-37` (`toast`) | EXISTS |
| **Router registry of ALL host routers (not just per-GPU resource blocks)** | -- | MISSING |
| **`MODELS_LIST` declared on the real routers** | -- | MISSING |
| **Router-discovery poll -> aggregated `_endpoint_catalog`** | -- | MISSING |
| **`GET /api/endpoints` route** | -- | MISSING |
| **`gpufarm endpoints` CLI command** | -- | MISSING |
| **`/dashboard/endpoints` panel + `_endpoints.html.j2` + COPY action** | -- | MISSING |

Net-new footprint for the read-only slice: roughly 350 LOC across
`coordinator.py`, `api.py`, `cli.py`, `dashboard.py`, one new template, and
small additions to `app.js` plus `resources.yaml`.

## 3. Router registry schema

The per-resource `router:` block already exists, but it is GPU-scoped: three
resources (`blackwell-host`, `blackwell-gpu0`, `blackwell-gpu1`) all point at
the same `http://kebab-rtx6000.lan:8080`, and two (`rtx5090`,
`mini-beast-8060s`) both point at `http://mini-beast.lan:8080`. A naive "poll
every resource.router" would hit each physical router 2-3x per tick and emit
duplicate catalog rows. We need a host-deduplicated registry.

Decision: add a NEW top-level `routers:` block in `resources.yaml`, one entry
per physical router, AND keep the existing per-resource `router:` blocks
unchanged (they still drive `_refresh_router_conflicts` and HostOps). The
catalog poll iterates `routers:` (deduplicated by host); the conflict refresh
keeps iterating `resources[*].router` (GPU-scoped). This is additive and
backwards compatible: manifests with no `routers:` block load unchanged and the
catalog is simply empty.

Why a top-level block rather than extending the resource blocks: mini-beast has
two resources (`rtx5090`, `mini-beast-8060s`) behind ONE router, so the router
is a host fact, not a GPU fact. A top-level list de-duplicates that cleanly and
gives one home for router-only metadata (`models_path`, `backends_path`,
`kind`) that does not belong on a GPU resource.

```yaml
# resources.yaml -- new top-level block (additive; absence => empty catalog)
routers:
  - host: kebab-rtx6000.lan
    url: http://kebab-rtx6000.lan:8080
    kind: fastapi                 # informational; how /v1/models behaves
    health_path: /health          # NOT /healthz, NOT /v1/health
    models_path: /v1/models       # DYNAMIC: reports actually-served ids
    admin_status_path: /admin/models/status
    capabilities: [health, models_list, admin_models_load, admin_models_unload, admin_models_status]
    # Two Blackwell PRO 6000 GPUs behind this router (cuda 0 language, 1 vision/ocr/tts).
    resources: [blackwell-gpu0, blackwell-gpu1]

  - host: mini-beast.lan
    url: http://mini-beast.lan:8080
    kind: python-router
    health_path: /health
    models_path: /v1/models       # STATIC list of 18 capability names
    backends_path: /backends      # live per-backend health+url (mini-beast only)
    admin_status_path: /admin/models/status
    capabilities: [health, stats, metrics, backends, models_list, admin_models_load, admin_models_unload, admin_models_status]
    # ONE router fronts BOTH the RTX 5090 (cuda 0) and the Strix Halo 8060S (Vulkan).
    resources: [rtx5090, mini-beast-8060s]

  # GB10 training nodes + mac-mini have no router. Recorded for completeness so
  # `gpufarm endpoints --all-hosts` can show "no router" rows. has_router=false.
  - host: kebab-spark.lan
    has_router: false
    note: GB10 training head; would need a FastAPI router stood up to serve inference.
  - host: kebab-gx10.lan
    has_router: false
  - host: kebab-gx10-2.lan
    has_router: false
  - host: kebab-gx10-3.lan
    has_router: false
  - host: mac-mini.lan
    has_router: false
    note: gpufarm coordinator/catalog host (dashboard :8765); not an inference router.
```

Loading: `manifests.py` gains a `_load_routers()` that parses this list into a
new frozen `RouterEntry` dataclass (mirroring `RouterConfig` in
`transports/types.py:91-114` but carrying `host`, `kind`, `models_path`,
`backends_path`, `admin_status_path`, `has_router`, `resources`). The
`models_list`/`backends` strings parse into existing `RouterCapability`
members; unknown strings raise at load time (the enum is fixed by design --
see `transports/types.py:18-32`). Entries with `has_router: false` are kept
(so `--all-hosts` can render them) but never polled.

Also: declare `models_list` (and on mini-beast, `backends`) on the existing
per-resource `router:` blocks so `RouterClient.has(MODELS_LIST)` returns True
there too. That is a one-line capability addition per block, no behavior change
for the conflict refresh.

## 4. Discovery + catalog

A new coordinator-owned poller mirrors `EndpointHealthMonitor`
(`endpoint_health_probe.py:143-373`) and the existing `_refresh_router_conflicts`
soft-fail pattern (`coordinator.py:2271-2356`).

### `RouterCatalogMonitor` (new, `router_catalog_probe.py`, ~150 LOC)

- Owns `_entries: dict[str, _RouterCatalogEntry]` keyed by router host, each
  holding `last_poll_ts`, `cached_rows`, `error`, `stale` flag.
- `async def poll_if_due(now)` throttled by `_router_catalog_refresh_ttl_sec`
  (default 30s -- tighter than the 300s conflict refresh at
  `coordinator.py:650`, because this is operator discovery, not safety
  critical, and routers swap models on demand so the view must be live).
- Per router with `has_router=true`:
  1. Construct `RouterClient(entry.as_router_config())`
     (`transports/router.py:79`).
  2. If `client.has(RouterCapability.MODELS_LIST)`, `await
     client.get(entry.models_path)` -> served names. On rtx6000 this is the
     dynamic actually-served list; on mini-beast it is the static 18-name list,
     so additionally, if `client.has(BACKENDS)`, `await
     client.get(entry.backends_path)` to learn live loaded/down state and
     per-backend groups.
  3. If `client.has(ADMIN_MODELS_STATUS)`, `await
     client.model_status(entry.admin_status_path)`
     (`transports/router.py:269`) for systemd-shaped loaded/unloaded/error
     state.
  4. Flatten to rows.
  5. `finally: await client.close()`.
- Fail-safe (mirrors `coordinator.py:2315-2326`): on `RouterUnreachable` /
  `RouterHttpError` / any exception, do NOT clear the cached rows. Instead mark
  the entry `stale=true`, set each of its rows' `status="stale"`, record the
  error string, and continue. A down router never crashes the tick and never
  silently empties the catalog -- the operator sees the last-known rows flagged
  STALE with a `last_probed` timestamp.

### Aggregated row shape

```python
{
  "model_id":     "qwen3-coder",                    # gpufarm/served id
  "served_names": ["qwen3-coder", "Qwen3-Coder-30B-A3B"],  # from /v1/models
  "host":         "mini-beast.lan",
  "router_url":   "http://mini-beast.lan:8080",
  "base_url":     "http://mini-beast.lan:8080/v1",  # curl-ready; what clients hit
  "status":       "loaded",   # loaded | unloaded | error | stale | unknown
  "kind":         "python-router",
  "last_probed":  "2026-06-14T19:40:12Z",
}
```

`base_url` is ALWAYS `http://HOST:8080/v1` (the router), never the internal
`127.0.0.1:80xx` backend URL. `status` is derived per row by merging
`/v1/models`, `/backends` (mini-beast), and `/admin/models/status`: a name that
appears served and shows `state=loaded` is `loaded`; one that admin reports
`unloaded` is `unloaded` (load-on-demand, e.g. the rtx6000 TTS backends);
`error` from admin maps to `error`; a whole router that failed to poll marks
all its rows `stale`.

### Coordinator wiring

- Construct `self._router_catalog = RouterCatalogMonitor(...)` next to
  `self._endpoint_health` (`coordinator.py:522-526`).
- Add `self._endpoint_catalog: list[dict] = []` cache next to
  `self._endpoint_health_cache` (`coordinator.py:526`).
- In `_tick` (`coordinator.py:966-991`), after the existing
  `self._endpoint_health.poll_if_due(now)` / `snapshot()` pair (lines 985-991),
  add:
  ```python
  await self._router_catalog.poll_if_due(now)
  self._endpoint_catalog = self._router_catalog.snapshot()
  ```
  The `_router_catalog.poll_if_due` is internally throttled, so it is cheap on
  ticks where it is not due (every 10s tick, real poll every 30s).

## 5. CLI: `gpufarm endpoints`

HTTP-online command, following `status` (`cli.py:811-871`), `queue`
(`cli.py:1174-1186`), and `rogue` (`cli.py:1255-1267`) exactly: call
`_coordinator_request` (`cli.py:791-808`), branch on status code, render a Rich
table, support `--json`.

```python
@main.command()
@click.option("--json", "emit_json", is_flag=True, help="Machine-readable output")
@click.option("--model", "model_filter", default=None, help="Filter rows by model_id / served name substring")
@click.option("--curl", "curl_model", default=None, help="Print a ready curl snippet for the named model and exit")
@click.option("--all-hosts", is_flag=True, help="Include hosts with no router (GB10 nodes, mac-mini)")
@click.pass_context
def endpoints(ctx, emit_json, model_filter, curl_model, all_hosts):
    """List reachable inference endpoints across the fleet.

    Talks to ``GET /api/endpoints`` on the coordinator, which aggregates each
    host router's ``/v1/models`` (+ mini-beast ``/backends``) into client-ready
    rows. ``base_url`` is the curl-ready router URL clients hit.
    """
    resp = _coordinator_request(ctx, "GET", "/api/endpoints")
    if resp.status_code != 200:
        click.echo(f"endpoints: HTTP {resp.status_code}: {resp.text}", err=True)
        sys.exit(1)
    rows = resp.json()["endpoints"]
    # --model filter; --all-hosts toggles inclusion of has_router=false rows.
    ...
    if curl_model:
        # resolve curl_model -> the loaded row's base_url, print snippet, exit 0
        # (exit 4 if not found / not loaded, so scripts can branch)
        click.echo(
            f"curl {base_url}/chat/completions \\\n"
            f"  -H 'Content-Type: application/json' \\\n"
            f"  -d '{{\"model\": \"{served}\", "
            f"\"messages\": [{{\"role\": \"user\", \"content\": \"hi\"}}]}}'"
        )
        return
    if emit_json:
        _print_json(rows); return
    # Rich table: model | host | base_url | served-names | status
```

Table columns: `model | host | base_url | served-names | status`. `status` is
color-coded the same way `rogue`/`status` color their state cells. Placement:
right after the `models` group (`cli.py:609-776`), around line 777, so
`gpufarm models ...` (offline manifest view) and `gpufarm endpoints` (live
fleet view) sit together.

`--curl <model>` prints a copy-paste curl against
`<base_url>/chat/completions` with the model's served name pre-filled -- this
is the CLI analogue of the web UI COPY button. Exit code 4 when the model is
not currently loaded anywhere, so a script can detect "not available."

## 6. Web UI: endpoints panel

Follows the established dashboard pattern: a view-builder, an HTMX-polled
partial, a route, and a delegated click handler.

- **`dashboard.py`**: add `_build_endpoints_view(coordinator)` next to
  `_build_health_view` (`dashboard.py:949`). It reads
  `coordinator._endpoint_catalog` (already populated each tick by section 4)
  and returns the rows as-is (optionally sorted host then model). Add the route
  after `dashboard_health` (`dashboard.py:1272-1285`), before
  `dashboard_reservations_history`:
  ```python
  @app.get("/dashboard/endpoints", response_class=HTMLResponse,
           name="dashboard_endpoints_partial")
  def dashboard_endpoints(request: Request) -> HTMLResponse:
      return templates.TemplateResponse(
          request, "_endpoints.html.j2",
          {"endpoints": _build_endpoints_view(coordinator),
           "now": _fmt_ts(datetime.now(UTC))},
      )
  ```
  Also add `"endpoints": _build_endpoints_view(coordinator)` to the
  `dashboard_index` context dict (`dashboard.py:1116`) for first paint.

- **`templates/index.html.j2`**: insert an ENDPOINTS `<section class="panel">`
  after the HEALTH panel, polling every 15s (`hx-get="/dashboard/endpoints"
  hx-trigger="every 15s, refresh"`). Endpoints change only on operator action,
  so 15s is plenty.

- **`templates/_endpoints.html.j2`** (new): a `data-table` with columns model,
  host, base_url, served_names, status badge, and an action cell holding a COPY
  button that mirrors the INSPECT/RELEASE buttons in `_whats_running.html.j2`:
  ```jinja2
  <button class="btn btn-xs" data-action="copy"
          data-copy-text="{{ row.base_url }}"
          title="Copy base_url to clipboard">COPY</button>
  ```
  A second `data-action="copy"` button can carry a full curl snippet in
  `data-copy-text` for one-click "copy a working request."

- **`static/app.js`**: add a `copy` case to the existing delegated
  `button[data-action]` handler (`static/app.js:363-388`), reusing the existing
  `toast()` (`static/app.js:28-37`):
  ```javascript
  if (a === "copy") {
    var txt = btn.getAttribute("data-copy-text");
    if (!txt) return;
    navigator.clipboard.writeText(txt)
      .then(function () { toast("copied: " + txt, "ok"); })
      .catch(function (err) { toast("copy failed: " + err, "err"); });
    return;
  }
  ```

- **`static/dashboard.css`**: optional `.badge.endpoint-loaded/unloaded/error/stale`
  classes reusing existing `--ok`/`--warn`/`--ink-dim` vars. No new button CSS
  needed (`.btn .btn-xs` already exist).

## 7. Deeper client/server awareness (the "go further")

Phase 2. Each item notes what the real routers actually support today (from the
survey), so we build only on real seams.

(a) **Registration on reserve/deploy.** Both routers already implement a
gpufarm-shaped admin API: `POST /admin/models/load`, `POST
/admin/models/unload`, `GET /admin/models/status`, accepting
`{"model_id":..., "served_name":...}` (legacy `{"model":...}` accepted too) --
see `transports/router.py:246-269` (`load_model`/`unload_model`/`model_status`)
and the resources.yaml capability declarations. So when gpufarm grants a
reservation that launches a model, it can call the host router's
`/admin/models/load` to make the router swap that backend in (load-on-demand is
already how both routers work). gpufarm's catalog (section 4) becomes the
source of truth: the coordinator owns "what should be loaded where," and the
admin calls drive the router to match. There is no `/admin/register` endpoint
on either router today -- registration means "load via the existing admin API,"
not "teach the router a new URL." That keeps Phase 2 on a seam that already
exists.

(b) **Fleet-wide resolve: `gpufarm route <model>`.** A CLI command (and `GET
/api/route?model=...`) that resolves a model name to its live `base_url` across
the whole fleet, so a client asks gpufarm once instead of knowing which host
runs what. It reads the same `_endpoint_catalog`, picks the row whose
`served_names` match and `status=loaded`, and prints just the `base_url` (or
exits non-zero if nothing serves it). This is the natural "gpufarm as the
unified front" primitive and is cheap once section 4 exists -- it is really
`gpufarm endpoints --curl` reduced to one URL. Optional later: gpufarm itself
listens on a stable port and reverse-proxies `/v1/*` to the resolved router, so
clients hardcode one gpufarm URL forever. That proxy is explicitly deferred (it
makes gpufarm a data-plane hop, with all the latency/availability weight that
implies); resolve-only keeps gpufarm in the control plane.

(c) **Bidirectional health.** `EndpointHealthMonitor`
(`endpoint_health_probe.py:143-373`) already probes live reservation endpoints
for model drift. Extend the catalog status so that when the router-served
probe fails, the row flips to `error`/`stale` (section 4 already does this for
a whole-router failure; (c) adds per-model granularity by cross-checking the
health probe result against the catalog row). The dashboard badge and CLI
status column then reflect "served but unhealthy," not just "router up."

Phase 2 is gated behind Phase 1 shipping and behind operator answers to the
open questions (section 9), because registration touches the live serving path.

## 8. Phased delivery

### Phase 1 -- read-only catalog (the shippable first slice)

Registry + discovery + catalog + CLI + UI panel. No writes to any router. Fully
backwards compatible: a manifest with no `routers:` block loads unchanged and
the catalog is empty (CLI prints "no endpoints," panel shows the empty row).

File-level changes:

| File | Change |
|---|---|
| `gpufarm/resources.yaml` | Add top-level `routers:` block (section 3); add `models_list` (+ `backends` on mini-beast) to the existing per-resource `router:` capability lists. |
| `gpufarm/transports/types.py` | Add `RouterEntry` dataclass (host, kind, paths, capabilities, has_router, resources) + `as_router_config()`. |
| `gpufarm/manifests.py` | `_load_routers()` parsing the new block into `RouterEntry`; expose `manifests.routers`. |
| `gpufarm/router_catalog_probe.py` (new) | `RouterCatalogMonitor` + `_RouterCatalogEntry`; `poll_if_due` / `snapshot`; soft-fail per router (mirror `coordinator.py:2315-2326`). |
| `gpufarm/coordinator.py` | Construct `_router_catalog` near line 522-526; add `_endpoint_catalog` cache near 526; call `poll_if_due` + `snapshot` in `_tick` after line 991; add `_router_catalog_refresh_ttl_sec=30.0` near line 650. |
| `gpufarm/api.py` | `GET /api/endpoints` returning `{"endpoints": coordinator._endpoint_catalog}`, placed after `/api/rogue` (`api.py:924`). |
| `gpufarm/cli.py` | `endpoints` command after the `models` group (`cli.py:776`), with `--json` / `--model` / `--curl` / `--all-hosts`. |
| `gpufarm/dashboard.py` | `_build_endpoints_view` near line 949; `/dashboard/endpoints` route after line 1285; add to `dashboard_index` context near line 1116. |
| `gpufarm/templates/index.html.j2` | ENDPOINTS panel after the HEALTH panel. |
| `gpufarm/templates/_endpoints.html.j2` (new) | Table + COPY buttons. |
| `gpufarm/static/app.js` | `copy` case in the delegated handler (after `static/app.js:375`). |
| `gpufarm/static/dashboard.css` | Optional `.badge.endpoint-*` classes. |

Test strategy (Phase 1):
- Unit: `RouterCatalogMonitor` against `httpx.MockTransport` via
  `RouterClient.install_transport` (the seam noted in
  `transports/router.py:93-95`). Cases: rtx6000 dynamic `/v1/models`,
  mini-beast static `/v1/models` + `/backends` merge, admin status merge,
  unreachable router -> rows marked `stale` and NOT cleared, malformed JSON ->
  soft-fail.
- Unit: `manifests.py` loads the new `routers:` block; a manifest with NO
  `routers:` block loads and yields an empty catalog (backwards-compat test).
- Unit: `RouterEntry.as_router_config()` round-trips capabilities into the
  fixed enum and raises on an unknown capability string.
- API: TestClient asserts `GET /api/endpoints` returns the coordinator's
  `_endpoint_catalog` shape.
- CLI: invoke `endpoints --json` against a fake coordinator; assert `--curl`
  prints a well-formed snippet and exits 4 when the model is not loaded.
- Dashboard: render `_endpoints.html.j2` with sample rows; assert COPY buttons
  carry `data-copy-text` and the empty-state row renders.

### Phase 2 -- registration + routing + bidirectional health

| File | Change |
|---|---|
| `gpufarm/coordinator.py` | On reservation grant/launch, call router `load_model` (`transports/router.py:246`); on release, `unload_model`. Cross-check `EndpointHealthMonitor` results into catalog status. |
| `gpufarm/api.py` | `GET /api/route?model=...` resolving model -> single `base_url`. |
| `gpufarm/cli.py` | `gpufarm route <model>` printing the resolved `base_url` (exit non-zero if none loaded). |
| `gpufarm/router_catalog_probe.py` | Per-model health flip (section 7c). |

Test strategy (Phase 2):
- Mock-transport tests for load/unload admin calls (assert `{"model_id","served_name"}` body, legacy fallback).
- `route` resolution: ambiguous (multi-host) and not-loaded cases.
- Health flip: a failing health probe flips the matching catalog row to `error`.

## 9. Open questions for the operator

1. Top-level `routers:` block vs. extending the per-resource `router:` blocks:
   the design proposes a new top-level block because mini-beast fronts two
   resources (rtx5090 + Strix Halo) with one router. Confirm you want the new
   block rather than inferring routers from the existing GPU-scoped blocks.
2. Phase 2 registration writes to the live serving path
   (`/admin/models/load` swaps backends). Should gpufarm be allowed to
   load/unload models on the routers as a side effect of reservations, or stay
   read-only and leave load/unload to you / the existing systemd units?
3. For `gpufarm route <model>`: resolve-only (gpufarm prints the host router's
   base_url, client connects directly) is the default. Do you also want the
   deferred reverse-proxy mode where gpufarm itself terminates `/v1/*` on a
   stable port and forwards, so clients only ever know one gpufarm URL?
4. mini-beast's `/v1/models` is a static 18-name capability list; the live
   loaded set comes from `/backends` + `/admin/models/status`. Should the
   catalog show all 18 advertised capabilities (with most `unloaded`), or only
   the backends currently loaded? This changes how many rows the panel shows
   for mini-beast.

# OpenMythos -> gpufarm cutover HOWTO

Operator runbook for flipping OpenMythos off the legacy `training/queue_*.sh`
shell loop and onto the `gpufarm` coordinator daemon. This file is the
last-mile follow-up to `gpufarm/docs/cutover_plan.md` (PR-8): the install
side is already done in shadow mode (see "Shadow-mode preinstall snapshot"
at the bottom); this doc covers the manual steps the operator runs when the
in-flight backfills finish.

The cutover is a one-way flip. Rollback is documented in step 7.

## Pre-flight (read-only)

These steps poke the system, they do not change anything. Run them first to
confirm the shadow install is intact.

1. SSH to the operator workstation that hosts the daemon:

       ssh alexm@kebab-spark.lan

2. Confirm the package and unit are in place:

       /home/alexm/.local/bin/gpufarm --help | head -3
       systemctl --user list-unit-files gpufarm-coordinator.service
       cat ~/.config/gpufarm/gpufarm.env

   Expected: CLI reports "Usage: gpufarm ...", unit shows
   `disabled / enabled` (installed but not enabled), env file lists
   `GPUFARM_MANIFEST_DIR`, `GPUFARM_REPO_ROOT`, `GPUFARM_STATE_DB`.

3. Confirm the state DB exists and was bootstrapped:

       sqlite3 ~/.local/state/gpufarm/state.sqlite \
           'SELECT status, COUNT(*) FROM runs GROUP BY status;'

   Expected: a single line `completed|<N>` where N is the number of
   already-finished (round, job) pairs (69 at install time).

4. Cross-check the gap report matches the bootstrap's leftover count:

       GPUFARM_MANIFEST_DIR=/home/alexm/OpenMythos/gpufarm \
           /home/alexm/.local/bin/gpufarm gaps --plan

   Expected: a non-empty gap table. The count should equal the
   `gaps_left_for_daemon` number printed when the bootstrap was run
   (16 at install time).

## Step 1: confirm the legacy backfill has drained

The cutover MUST NOT happen while the legacy GPU-0 / GPU-1 SFT and eval
backfills are still running -- the daemon would race them for the same
checkpoints. Wait for both to finish before proceeding.

    ssh alexm@kebab-spark.lan
    pgrep -af 'gpu0_sft_backfill_v2|gpu1_backfill_v2' || echo done

The expected ETA at install time was ~22:00-23:30 EDT on 2026-05-14. The
phrase `done` alone (with no `pgrep` matches) confirms both backfills have
exited and `kebab-rtx-vllm.service` is back up (the gpu0 backfill restarts
the vLLM service on EXIT).

If either process is still listed, **stop** and rerun this step later.
DO NOT kill the backfills.

## Step 2: enable + start the gpufarm daemon

From the operator workstation:

    systemctl --user enable --now gpufarm-coordinator
    systemctl --user status gpufarm-coordinator --no-pager

Expected: `Active: active (running)`. The unit logs go to the journal
(`journalctl --user -u gpufarm-coordinator -f`).

If the daemon fails to start, check:

* env file syntax (`EnvironmentFile=` only accepts bare `KEY=VALUE` lines,
  no quotes / no sections / no arrays);
* manifest dir actually contains `{resources,jobs,rounds}.yaml`;
* the user can write to `~/.local/state/gpufarm/`.

## Step 3: confirm the daemon is healthy

    curl -s http://127.0.0.1:8765/health
    /home/alexm/.local/bin/gpufarm status

`/health` returns JSON with `"status":"ok"` (or similar). `gpufarm status`
prints a snapshot of GPUs + run queue (no "coordinator unreachable" error).

If you want the dashboard:

    /home/alexm/.local/bin/gpufarm dashboard --open

(loopback-only HTTP; safe to leave running).

## Step 4: submit any remaining gaps

The bootstrap already populated `state.sqlite` with the COMPLETED rows for
work that finished pre-cutover, so the daemon only sees the real gaps.
Push them all in one shot:

    /home/alexm/.local/bin/gpufarm submit --all-gaps

The coordinator decides which gaps go to which resource_class based on the
job manifest. Watch progress with `gpufarm status` or
`journalctl --user -u gpufarm-coordinator -f`.

For a targeted subset:

    /home/alexm/.local/bin/gpufarm submit --round r29           # one round
    /home/alexm/.local/bin/gpufarm submit sft_lora --round r24  # one (round, job)

## Step 5: retire the legacy `queue_*.sh` scripts

On the OPERATOR workstation (where this repo is editable), move the legacy
queue scripts out of the way. Disable rather than delete so a rollback
keeps them recoverable from one place:

    cd ~/git/OpenMythos
    mkdir -p training/legacy
    for f in training/queue_*.sh; do
        git mv "$f" "training/legacy/$(basename "$f").disabled"
    done
    git commit -m "[refactor][cutover][move legacy queue_*.sh to training/legacy/]"

Commit reaches the kyegomez/OpenMythos fork; mirror to your personal
fork as usual.

## Step 6: retire the legacy `auto_eval_round*.sh` watchers

The auto-eval watchers polled the cluster log line `auto_eval_roundN
pipeline complete` to fence rounds. The daemon's tick loop replaces that
fence (it sees a round's `train_round` Run flip to COMPLETED and dispatches
the eval bundle automatically once round entries are in `rounds.yaml`).

    cd ~/git/OpenMythos
    for f in training/auto_eval_round*.sh; do
        git mv "$f" "training/legacy/$(basename "$f").disabled"
    done
    git commit -m "[refactor][cutover][move legacy auto_eval_round*.sh to training/legacy/]"

DO NOT move `training/retry_cluster_training.sh`; it is the NCCL-hang
recovery hook still referenced from `Resource.recovery_hook` in
`gpufarm/resources.yaml`.

DO NOT move `training/auto_paper_integrate*.py` / `.sh`; that pipeline is
downstream of gpufarm and unaffected by the cutover.

## Step 7: rollback (only if cutover misbehaves)

If you have to back out:

1. Stop and disable the daemon:

       systemctl --user disable --now gpufarm-coordinator

2. Restore the shell scripts. Either from the legacy directory or from git
   history -- pick whichever is shorter for what failed.

       cd ~/git/OpenMythos
       for f in training/legacy/queue_*.sh.disabled; do
           name=$(basename "$f" .disabled)
           git mv "$f" "training/$name"
       done
       # repeat the loop for auto_eval_round*.sh.disabled if you moved them
       git commit -m "[rollback][cutover][restore legacy queue_*.sh]"

3. The state.sqlite stays on disk through the rollback. A second cutover
   attempt later picks up where this one left off; nothing wipes the
   COMPLETED rows the bootstrap inserted.

4. The vLLM service on RTX6000 is unaffected by the daemon's lifecycle --
   it stays in whatever state it was in before the cutover.

## Shadow-mode preinstall snapshot

For reference, this is what the shadow-mode install left on
`kebab-spark.lan` (2026-05-14):

| Component                | Path                                              | Status                      |
|--------------------------|---------------------------------------------------|-----------------------------|
| gpufarm source           | `/home/alexm/gpufarm` (rsync from local)          | installed (v0.3.0.dev0)     |
| gpufarm CLI              | `/home/alexm/.local/bin/gpufarm`                  | working, `--help` OK        |
| Env file                 | `/home/alexm/.config/gpufarm/gpufarm.env`         | placed                      |
| Systemd unit             | `/home/alexm/.config/systemd/user/gpufarm-coordinator.service` | installed, disabled (NOT started) |
| `daemon-reload`          | n/a                                               | ran once                    |
| State DB                 | `/home/alexm/.local/state/gpufarm/state.sqlite`   | bootstrapped (69 COMPLETED rows) |
| Daemon process           | n/a                                               | NOT running                 |
| Gaps awaiting dispatch   | `gpufarm gaps --plan`                             | 16 entries                  |

The legacy SFT backfills were still running at install time:
`gpu0_sft_backfill_v2.sh` (PID 2915868) and `gpu1_backfill_v2.sh` (PID
2916064) on spark. Step 1 above blocks the cutover until both finish.

## Why not enable the daemon during shadow install?

The shadow install deliberately stops one step short of `systemctl --user
enable --now gpufarm-coordinator` because:

1. Two simultaneous schedulers (the legacy `queue_*.sh` loop AND the
   gpufarm coordinator) would race the same checkpoints.
2. The gpu0 backfill restarts `kebab-rtx-vllm.service` on EXIT. Starting
   the daemon before that EXIT can cause the daemon to try dispatching
   work onto a host whose vLLM service is missing.
3. Cutover-day responsibility for "press the button" is the operator's,
   not the install agent's. Shadow mode lets the install be reviewed in
   peace before anything moves.

## Bootstrap re-run

`tools/gpufarm_bootstrap_from_disk.py` is idempotent. Re-run it any time
to absorb newly-completed artifacts that were produced outside the daemon
(e.g., another in-flight backfill that completes after step 1 but before
step 2). It will skip rows already in the DB and only insert new ones.

    python3 tools/gpufarm_bootstrap_from_disk.py \
        --manifest-dir /home/alexm/OpenMythos/gpufarm \
        --state-db /home/alexm/.local/state/gpufarm/state.sqlite

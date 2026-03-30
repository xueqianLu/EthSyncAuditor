# Logical Synchronization Graph (LSG) Schema Specification

Version: 1  
Generated at: 2026-03-08T07:58:08Z

This document defines a **common, machine-readable schema** for Logical Synchronization Graphs (LSGs)
that all client agents (Lighthouse, Prysm, Teku, Grandine, etc.) MUST use when describing
high-level workflows for comparison.

The corresponding on-disk representation is a single YAML file per client, typically named:

- `LSG_<ClientName>_Sync.yml` (or more generally `LSG_<ClientName>.yml`)

All clients MUST adhere to the structure and naming rules in this spec so that LSGs can be
compared automatically.

---

## 1. Top-Level YAML Structure

```yaml
version: <int>          # Schema version. For this spec: 1
client: <string>        # Client identifier, e.g. "prysm", "lighthouse", "teku", "grandine"
generated_at: <string>  # RFC3339 timestamp; informational only, ignored for diff/comparison

guards:                 # Global guard vocabulary (Σ)
  - name: <GuardName>
    category: <string>      # e.g. network|time|validation|mode|exec|validator|state
    description: <string>   # Human-readable meaning

actions:                # Global action vocabulary (Γ)
  - name: <ActionName>
    category: <string>      # e.g. network|sync|block|peer|schedule|exec|validator
    description: <string>   # Human-readable meaning

workflows:              # Set of LSGs (G_i), one per business workflow
  - id: <WorkflowId>        # Stable identifier shared across clients
    name: <string>          # Human-readable name
    description: <string>
    mode: <string>          # Short tag, e.g. InitialSync, RegularSync
    initial_state: <StateId>
    states:
      - id: <StateId>       # Unique within this workflow, recommended: prefix.phase
        label: <string>     # Human-readable label
        category: <string>  # e.g. init|peer_select|request|wait|validate|import|peer|recover|progress|terminal
        transitions:
          - guard: <GuardName|TRUE>
            actions: [<ActionName>, ...]
            next_state: <StateId>
            evidence:               # OPTIONAL (recommended in concrete client files)
              file: <string>        # Relative path to source file, e.g. beacon-chain/sync/service.go
              function: <string>    # Function or method name, e.g. "(s *Service) runInitialSync"
              lines: [<int>, <int>] # [start_line, end_line], 1-based inclusive
```

### 1.1. Required vs Optional Fields

- `version`, `client`, `guards`, `actions`, `workflows` are REQUIRED.  
- `generated_at` is OPTIONAL but RECOMMENDED; comparison tools MUST ignore differences in this field.  
- Within `workflows[*].states[*].transitions[*]`, `evidence` is OPTIONAL and may be omitted when
  the mapping to code is not yet stabilized.

---

## 2. Global Guard Vocabulary (guards)

`guards` define the **conditions** that govern state transitions (Σ).  
All agents SHOULD reuse the same guard names wherever possible so workflows
can be compared across different clients.

Example guards (non-exhaustive, but RECOMMENDED baseline set):

```yaml
guards:
  # Network / input
  - name: RespRecv
    category: network
    description: Response received from a peer (range, checkpoint, blob, etc.).
  - name: RespInvalid
    category: network
    description: Response or payload structurally or cryptographically invalid.
  - name: GossipRecvBlock
    category: network
    description: New block received via gossip.
  - name: GossipRecvAttestation
    category: network
    description: New attestation received via gossip.
  - name: PeerDisconnected
    category: network
    description: Peer disconnected during or before a request/response.
  - name: NewPeerAvailable
    category: network
    description: At least one eligible peer is available for selection.

  # Time / retry
  - name: TimeoutExpired
    category: time
    description: Hard timeout for a pending network or execution request has fired.
  - name: BackoffExpired
    category: time
    description: Previously scheduled backoff delay has elapsed.

  # Validation / state
  - name: MissingParent
    category: validation
    description: Block references a parent that is not yet known/imported.
  - name: AlreadyKnown
    category: validation
    description: Block or attestation already known or imported.
  - name: ForkChoiceReject
    category: validation
    description: Fork-choice rule rejects importing the candidate block.
  - name: ReachedTargetSlot
    category: state
    description: Local sync slot has reached or exceeded the current target slot.

  # Mode / configuration
  - name: ModeIsInitialSync
    category: mode
    description: Node is in initial sync mode (pre-regular sync, including backfill).
  - name: ModeIsRegularSync
    category: mode
    description: Node is in regular gossip-based sync mode.
  - name: ModeIsCheckpointSync
    category: mode
    description: Node is in checkpoint-based bootstrap sync mode.
  - name: ModeIsOptimistic
    category: mode
    description: Node performs optimistic execution payload import.

  # Execution-layer related
  - name: ExecutionClientSyncing
    category: exec
    description: Execution client is not fully synced; payloads may be imported optimistically.
  - name: ExecutionValidationSucceeded
    category: exec
    description: Execution client reports successful validation of an execution payload.
  - name: ExecutionValidationFailed
    category: exec
    description: Execution client reports failed validation of an execution payload.

  # Validator / duties
  - name: HasProposerDuty
    category: validator
    description: Local validator has a block proposal duty in this slot.
  - name: HasAttesterDuty
    category: validator
    description: Local validator has an attestation duty in this slot.
  - name: SelectedAsAggregator
    category: validator
    description: Local validator has been selected as an aggregator for a committee.
```

Agents MAY extend this list with client-specific guards, but SHOULD prefer reusing
names from this baseline when the semantics match.

---

## 3. Global Action Vocabulary (actions)

`actions` define the **effects** of transitions (Γ).  
All agents SHOULD reuse the same action names wherever possible.

Example actions (non-exhaustive, but RECOMMENDED baseline set):

```yaml
actions:
  # Network
  - name: SendRangeRequest
    category: network
    description: Send BeaconBlocksByRange (or equivalent) request to a peer.
  - name: SendCheckpointRequest
    category: network
    description: Request blocks or states around a finalized checkpoint.
  - name: SendStatus
    category: network
    description: Send status/handshake message to peer.
  - name: SendGoodbye
    category: network
    description: Send a goodbye message before disconnecting.
  - name: DisconnectPeer
    category: network
    description: Disconnect and optionally blacklist a peer.
  - name: SubscribeGossip
    category: network
    description: Subscribe to gossip topics required for regular sync.

  # Sync control
  - name: UpdateSyncTarget
    category: sync
    description: Update internal sync target slot or root from peer information.
  - name: BuildPeerQueue
    category: sync
    description: Build or refresh priority queue of sync peers.
  - name: EnterRegularSync
    category: sync
    description: Switch internal state machine into regular sync mode.

  # Block / batch processing
  - name: ValidateBatch
    category: block
    description: Validate a batch of blocks from a range or checkpoint response.
  - name: ApplyBatch
    category: block
    description: Apply a batch of blocks and advance state/fork-choice.
  - name: StoreBlock
    category: block
    description: Persist block without necessarily making it canonical head.
  - name: ValidateBlock
    category: block
    description: Validate an individual block against consensus rules.
  - name: ApplyBlock
    category: block
    description: Apply a single block and update state.
  - name: UpdateForkChoice
    category: block
    description: Update fork-choice structures and canonical head.
  - name: MarkBlockInvalid
    category: block
    description: Mark block(s) as invalid in local store.
  - name: RequestParents
    category: block
    description: Request parent blocks needed to process a child.

  # Peer management
  - name: PenalizePeer
    category: peer
    description: Decrease peer reputation score due to misbehavior or timeout.
  - name: UpdatePeerScore
    category: peer
    description: Recompute or persist updated peer score.
  - name: SelectNextPeer
    category: peer
    description: Select next peer from queue according to scoring heuristics.

  # Scheduling / retry
  - name: ScheduleTimeout
    category: schedule
    description: Schedule hard timeout for a request or operation.
  - name: ScheduleBackoff
    category: schedule
    description: Schedule a backoff delay before retrying an operation.

  # Execution / optimistic
  - name: ApplyOptimisticBlock
    category: exec
    description: Apply block with execution payload without waiting for full EL validation.
  - name: MarkPayloadPending
    category: exec
    description: Record payload awaiting execution-layer validation.
  - name: TriggerExecutionValidation
    category: exec
    description: Ask execution client to validate one or more pending payloads.
  - name: RollbackToSafeHead
    category: exec
    description: Roll back canonical head to last safe (finalized/justified) point.
  - name: MarkPayloadInvalid
    category: exec
    description: Mark execution payload as invalid and avoid re-import.

  # Validator / attestation / block production
  - name: FetchDuties
    category: validator
    description: Query beacon node for validator duties for upcoming slots/epochs.
  - name: BuildAttestation
    category: validator
    description: Construct attestation using committee assignment and head state.
  - name: SignAttestation
    category: validator
    description: Sign attestation with validator key.
  - name: PublishAttestation
    category: validator
    description: Submit attestation to beacon node for gossip/broadcast.
  - name: BuildBlock
    category: validator
    description: Construct block (with execution payload if applicable) for a slot.
  - name: SignBlock
    category: validator
    description: Sign block with proposer key.
  - name: PublishBlock
    category: validator
    description: Submit signed block to beacon node for gossip and import.
  - name: ComputeAggregate
    category: validator
    description: Aggregate attestations or sync committee messages.
  - name: SignAggregate
    category: validator
    description: Sign aggregated attestation or sync aggregate.
  - name: PublishAggregate
    category: validator
    description: Submit signed aggregate to beacon node.
```

Again, agents MAY extend this with client-specific actions, but SHOULD reuse
baseline names when semantics align.

---

## 4. Workflows (business-level LSGs)

Each entry in `workflows` represents one high-level **business workflow**.
For cross-client comparison, the following seven `id` values are RESERVED and SHOULD
be implemented by all client agents:

1. `initial_sync`  
2. `regular_sync`  
3. `checkpoint_sync`  
4. `attestation_generate`  
5. `block_generate`  
6. `aggregate`  
7. `execute_layer_relation`  

### 4.1. Common fields

For each workflow:

```yaml
workflows:
  - id: <one-of-the-7-ids>
    name: <HumanReadableName>
    description: <Longer description of the workflow>
    mode: <ShortModeName>            # e.g. InitialSync, RegularSync, etc.
    initial_state: <StateId>
    states:
      - id: <StateId>
        label: <string>
        category: <string>
        transitions:
          - guard: <GuardName|TRUE>
            actions: [<ActionName>, ...]
            next_state: <StateId>
            evidence:        # OPTIONAL
              file: <string>
              function: <string>
              lines: [<int>, <int>]
```

### 4.2. State and Transition Naming

- `StateId` SHOULD be namespaced by workflow prefix for clarity, e.g. `initial.peer_select`, `regular.idle`.  
- `category` is free-form but SHOULD use a small, shared vocabulary where possible, such as:
  - `init`, `peer_select`, `request`, `wait`, `validate`, `import`, `recover`, `peer`, `progress`, `idle`, `receive`, `build`, `sign`, `publish`, `compute`, `error`, `terminal`.  
- `guard` MUST either be a known `GuardName` from `guards` or the special literal `TRUE` for
  unconditional transitions.  
- `actions` MUST be drawn from the `actions` list; it MAY be empty (`[]`).

### 4.3. Evidence (Optional, Client-Specific)

`evidence` allows mapping a transition back to concrete implementation code.

- It is OPTIONAL in this schema: early versions of LSGs may omit it.  
- When present, all three fields SHOULD be provided:
  - `file`: relative path from the repo root to the implementing file.  
  - `function`: function or method name implementing this transition.  
  - `lines`: a 2-element array `[start, end]` (1-based, inclusive) giving an approximate
    line range that implements the transition.

Example transition with evidence:

```yaml
    transitions:
      - guard: RespRecv
        actions: [ValidateBatch]
        next_state: initial.import_batch
        evidence:
          file: beacon-chain/sync/initial-sync/service.go
          function: "(s *Service) handleRangeResponse"
          lines: [120, 210]
```

Comparison tools MAY ignore `evidence` when comparing high-level structures,
using it only to backtrack to source code for debugging or further analysis.

---

## 5. Example: Minimal Initial Sync Workflow Skeleton

This is a **minimal**, schematic example of `initial_sync` in the standard format:

```yaml
version: 1
client: example-client
guards:
  - name: RespRecv
    category: network
    description: Response received from a peer.
  - name: TimeoutExpired
    category: time
    description: Timeout fired.
  - name: NewPeerAvailable
    category: network
    description: At least one peer is available.
  - name: ReachedTargetSlot
    category: state
    description: Reached target slot.

actions:
  - name: SendRangeRequest
    category: network
    description: Request a range of blocks.
  - name: ScheduleTimeout
    category: schedule
    description: Schedule a timeout.
  - name: ApplyBatch
    category: block
    description: Apply a batch of blocks.
  - name: UpdateForkChoice
    category: block
    description: Update fork-choice.

workflows:
  - id: initial_sync
    name: "Initial Synchronization"
    description: "Bootstrap from empty state to target slot."
    mode: InitialSync
    initial_state: initial.peer_select
    states:
      - id: initial.peer_select
        label: "Pick sync peer"
        category: peer_select
        transitions:
          - guard: NewPeerAvailable
            actions: []
            next_state: initial.request_range
      - id: initial.request_range
        label: "Request block range"
        category: request
        transitions:
          - guard: TRUE
            actions: [SendRangeRequest, ScheduleTimeout]
            next_state: initial.wait_response
      - id: initial.wait_response
        label: "Wait for response"
        category: wait
        transitions:
          - guard: RespRecv
            actions: [ApplyBatch, UpdateForkChoice]
            next_state: initial.check_progress
          - guard: TimeoutExpired
            actions: []
            next_state: initial.peer_select
      - id: initial.check_progress
        label: "Check progress"
        category: progress
        transitions:
          - guard: ReachedTargetSlot
            actions: []
            next_state: initial.done
          - guard: TRUE
            actions: []
            next_state: initial.peer_select
      - id: initial.done
        label: "Initial sync complete"
        category: terminal
        transitions: []
```

---

## 6. Usage Guidelines for Agents

1. **One schema, many clients**: all client agents MUST follow this spec so LSGs can be
   compared mechanically across implementations.  
2. **Shared vocabulary first**: before defining a new `guard` or `action`, agents SHOULD
   check whether an equivalent semantic already exists in this document and reuse its name.  
3. **Stable workflow IDs**: the seven reserved `workflows[*].id` values MUST NOT be changed
   or repurposed; client-specific workflows, if needed, should use additional IDs.  
4. **Evidence as refinement**: agents MAY first publish LSGs without `evidence` and
   incrementally add `evidence` entries as code mapping becomes precise.  
5. **Comparison tools**: tooling that compares LSGs across clients SHOULD treat:
   - `version`, `client`, `guards`, `actions`, `workflows[*].id`, `states`, `transitions`
     as comparison-relevant;  
   - `generated_at` and `evidence` as informational (ignored for structural diff).

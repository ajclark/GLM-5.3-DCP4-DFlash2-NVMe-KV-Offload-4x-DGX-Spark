# Fable coalesced reader review: completed findings

Fable reviewed the reader, short-read fixes, and asynchronous CUDA handoff on
2026-09-13. The final review found no blocking correctness defect in the examined
ownership, ordering, alias-accounting, and cancellation paths.

- Legal short direct reads now finish through a buffered descriptor for the same
  inode. An eager direct-I/O probe selects native fallback before consumption
  when the filesystem is unsupported. Both initial findings were addressed.
- The consumer waits on the upload event; backing storage is recorded on its
  stream. Each pinned tile is fenced before reuse. Cancellation drains pending
  reads and synchronizes uploads before releasing staging memory.
- Retained tensor views keep their complete batch charged against the owned-byte
  budget. Read jobs drain before the reader changes its file or source offset.
- `upload_wait_seconds` records the final upload wait, not cumulative per-batch
  waiting. A spinning host event wait does not demonstrate useful CPU work.

The [implementation report](../../docs/COALESCED-LOADER-IMPLEMENTATION.md) records
the subsequent sandbox, GPU parity, full serving, and durable-KV validation.

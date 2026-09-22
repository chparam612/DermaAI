# Training Module (`src/train`)

This module manages training loops, validation cycles, optimizer scheduling, and checkpoint serialization.

## Responsibilities

- **Epoch execution**: Batched forward/backward pass with gradient scaling (AMP).
- **Validation**: Periodic metric computation against validation splits.
- **Checkpointing**: Automatic saving of `best_model.pt` based on validation macro F1 or AUROC.
- **Early stopping**: Termination upon metric plateauing.

The complete training execution pipeline will be implemented in Phase 1 following model backbone integration.

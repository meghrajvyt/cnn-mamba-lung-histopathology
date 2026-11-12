# Development history

This repository was created for publication from the authors' private
development repository, which holds the complete history of the work.
Three rules in the paper were fixed before the results they govern existed:
the choice between the two residual hybrid variants, the LungHist700
small-data protocol and the protocol of the analyses added after the
pre-submission review. These are the development commits that fixed them,
with the first commit and the results of the added analyses for context
(author dates as recorded by git, Indian Standard Time):

| Commit | Date | What it fixed or added |
|---|---|---|
| `6f354c3` | 2026-09-11 21:02 +0530 | First commit of the development repository |
| `9195061` | 2026-10-10 21:58 +0530 | Rule for choosing between the two residual hybrid variants: seed-42 validation accuracy on LC25000, ties to hybrid-res (`scripts/queue_paper1.sh`) |
| `0a906ca` | 2026-10-11 20:56 +0530 | LungHist700 small-data protocol: 100 epochs without early stopping, learning rate per architecture chosen on fold-0 validation (`scripts/queue_lh700.sh`) |
| `83caa1d` | 2026-10-20 05:25 +0530 | Protocol of the analyses added after the pre-submission review (`docs/added_analyses_protocol.md`) |
| `361e813` | 2026-10-20 15:18 +0530 | Results of those analyses |

This release was exported from development commit `c0dac1f` (2026-11-12 07:48 +0530); the
export copies an explicit list of files without changing them. The dated
history of the development repository is available to editors and
reviewers on request.

# Running the study on AWS

> **Written on 2026-09-25, before the study design was final.** The guidance
> on the instance itself still applies: GPU quota, AMI, spot settings,
> auto-stop and billing alarm. The tier plan, the cost estimates and the run
> commands (`scripts/run_study.sh --tier`) do not. The paper's runs were made on one
> g5.xlarge (NVIDIA A10G) by the experiment queues described in `README.md`,
> under "Reproducing the study".

## Cost: roughly $20–55 for the whole study

Derived from measured throughput on this project, not from guesswork — but the
GPU figures are extrapolated from an M2 measurement, so treat them as estimates
until step 4 confirms them on the actual instance.

### What the study needs

| Tier | Runs | What it buys |
|---|---|---|
| 1 | 5 | Leakage comparison, single-pathway ablations, plain-CNN control |
| 2 | 3 | Fusion ablation, transformer baselines |
| 3 | 10 | Two extra seeds over tier 1, for mean ± std and paired tests |
| | **18** | |

Assuming 30 epochs per run (`patience 10` usually triggers before 50 — the M2 run
was already plateauing by epoch 5):

| Instance | GPU | Est. min/epoch | Study GPU-hours |
|---|---|---|---|
| g4dn.xlarge | T4 16 GB | ~12 | ~95 |
| g5.xlarge | A10G 24 GB | ~6 | ~50 |

### Cost

| Setup | Rate | Total |
|---|---|---|
| g4dn.xlarge on-demand | $0.526/hr | **~$50** |
| g4dn.xlarge **spot** | ~$0.19/hr | **~$18** |
| g5.xlarge on-demand | $1.01/hr | **~$50** |
| g5.xlarge **spot** | ~$0.35/hr | **~$18** |

Plus about $2 for a 100 GB gp3 volume over a week. Data transfer in is free, and
out is negligible (results are a few hundred MB).

On-demand rates verified against [AWS EC2 On-Demand Pricing](https://aws.amazon.com/ec2/pricing/on-demand/)
for us-east-1; g4dn.xlarge at $0.526/hr is [independently confirmed](https://markaicode.com/pricing/amazon-ec2-spot-vs-on-demand-gpu-cost-analysis/).
Spot rates fluctuate by region and time — check before launching. *Content
rephrased for compliance with licensing restrictions.*

**Tier 1 alone is ~21 GPU-hours, so about $4 on spot.** That is the minimum
credible submission, and it is cheap enough to do first and decide afterwards.

### Use spot instances

This codebase is unusually well suited to spot, because interruption handling was
already built and tested:

- checkpoints are written every epoch
- `scripts/run_study.sh` skips completed runs and resumes partial ones from
  `last_model.pth`
- the resume path restores optimiser, scheduler, scaler, best-so-far accuracy and
  early-stopping counters (two bugs here were found and fixed specifically so
  multi-session runs behave correctly)

A spot reclaim costs you at most one partial epoch. That turns a 65% discount into
essentially free money for this workload.

---

## The blocker nobody warns you about

**New AWS accounts have a GPU quota of zero.** Launching will fail with
`VcpuLimitExceeded` regardless of billing status, and this is by far the most
common reason a first attempt fails.

Request the increase **before** planning around a timeline:

1. Service Quotas → Amazon EC2
2. For on-demand: *Running On-Demand G and VT instances* → request **at least 4**
   vCPUs (g4dn.xlarge is 4 vCPUs)
3. For spot: *All G and VT Spot Instance Requests* → request **at least 4**
4. Approval takes anywhere from minutes to a couple of days

Also note the AWS free tier does **not** include GPU instances. If you are at a
university, ask about AWS research credits or an educational grant first — this
study is small enough that a modest credit covers it entirely.

---

## Setup

### 1. Launch

- **AMI**: Deep Learning OSS Nvidia Driver AMI (PyTorch) — CUDA and drivers
  preinstalled, which avoids the CPU-only-torch trap entirely
- **Instance**: `g4dn.xlarge` (cheapest viable) or `g5.xlarge` (about 2x faster
  for roughly the same total cost)
- **Storage**: 100 GB gp3
- **Spot**: set request type to spot, with interruption behaviour **Stop**, not
  Terminate, so the volume survives

### 2. Bootstrap

```bash
ssh -i key.pem ubuntu@<ip>
git clone https://github.com/meghrajvyt/cnn-mamba-lung-histopathology.git
cd cnn-mamba-lung-histopathology
bash scripts/aws_bootstrap.sh  # env, deps, data, verification
```

### 3. Verify the GPU is real

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

A `+cpu` suffix or `False` means the wrong torch build: install the CUDA build
from `requirements-lock.txt` with the index URL given in README.md
("Reproducing the study").

### 4. Size the run, then confirm the estimate

```bash
python3 tools/find_batch_size.py --all-archs --depth 4
```

On a 16 GB T4 expect batch 16–32 for `hybrid` with `--grad_checkpoint`. Then
measure one epoch before trusting the cost table above:

```bash
scripts/run_experiment.sh timing_probe --epochs 1 --batch_size 32 --mamba_depth 4 --grad_checkpoint
```

Multiply the observed epoch time by 30 × 18 to get your real study cost.

### 5. Run

```bash
tmux new -s study
nohup scripts/run_study.sh --tier 1 > logs/study_tier1.log 2>&1 &
```

`tmux` matters: an SSH drop otherwise kills the run.

### 6. Protect yourself from the classic mistake

The usual way to waste money here is leaving an idle instance running. Auto-stop
when the study finishes:

```bash
# append to the study command
(scripts/run_study.sh --tier 1; sudo shutdown -h now) &
```

Also set a billing alarm (Billing → Budgets) at, say, $20. And verify
`scripts/aws_bootstrap.sh` copied results off the box before you terminate anything.

### 7. Retrieve results

```bash
# from your laptop
scp -i key.pem -r ubuntu@<ip>:~/cnn-mamba-lung-histopathology/results ./aws_results
```

Then locally:

```bash
python3 aggregate_results.py --results-dirs ./aws_results/*/ --output-dir ./aws_results/_aggregate
python3 compare_models.py --all ./aws_results
```

---

## Cheaper alternatives

AWS is the most expensive of the sensible options. If institutional billing is not
a requirement:

| Provider | Typical rate | Note |
|---|---|---|
| Vast.ai | $0.15–0.40/hr | RTX 3090/4090, often faster than a T4 |
| RunPod | $0.20–0.50/hr | Simpler UX, per-second billing |
| Lambda Labs | $0.50–0.80/hr | A10/A100, good availability |

All are SSH-accessible, so `scripts/run_study.sh` works unchanged. On Vast.ai the full
study could be closer to $15.

Choose AWS if you have credits, need institutional invoicing, or want the quota
and support story. Choose Vast.ai or RunPod if you are paying personally and want
the lowest number.

---

## Honest caveats

The min/epoch figures are extrapolated from an M2 GPU measurement (2.23 img/s at
depth 4, 224px) scaled by memory bandwidth, because this model is
bandwidth-bound rather than compute-bound. A T4 has roughly 3x the M2's bandwidth
and, unlike MPS, benefits properly from large batches and tensor-core AMP. The
real number could reasonably be 1.5x better or worse than the table. Step 4 above
settles it in about ten minutes, and costs under a dollar.

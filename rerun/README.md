# Rerun of the leprosy diffusion model

This folder retrains the model from arXiv:2609.13226 v1 exactly as it was trained,
then measures it properly. It answers one question the original run could not:
how much did fine-tuning on leprosy images change what the model generates?

Nothing in the rest of the project is modified. The original notebook
(`model_v5_saved/v5-7.ipynb`) is the reference for every setting here.

## What is the same as v1, and what changed

| | v1 notebook | this rerun |
|---|---|---|
| Data, crops, mask features, context hint | as in `ROIWoundDatasetV3` | identical |
| Model, optimizer, steps, prompts, sampler | as in `Config` / `FineTuneConfig` | identical |
| Base model files | `runwayml/stable-diffusion-v1-5` (no longer on Hugging Face) | `stable-diffusion-v1-5/stable-diffusion-v1-5`, the mirror of the same weights |
| Base vs fine-tuned model | fine-tuning modified the base model object in place | each model is saved to and loaded from its own folder; every image set records which weights made it |
| Leprosy images for fine-tuning | all 764 | 80% (611); the other 153 are held out for evaluation |
| Masks used for evaluation | Kaggle wound masks | held-out leprosy masks (the Kaggle protocol is also rerun, to compare with v1) |
| Real reference images | raw images, colour cards still in | card-removed images, as used in fine-tuning |
| Per-pair scores | lost | saved, so every number gets a confidence interval |
| Interruptions | lose the run | checkpoints every 2,000 steps; rerun the command to continue |

Already checked on a Mac before any GPU time:
* `smoke_test.py` runs every step with tiny random models, including an interrupted run that resumes;
* the real UNet has 859,541,124 parameters, with 609,928,644 trainable during fine-tuning, the same as v1 printed;
* `prepare.py` on the real data finds 3,769 wound pairs (as in the paper), and its rebuild of the v1 real
  reference skips the same 8 empty masks the paper lists.

## Files

| file | purpose |
|---|---|
| `run_all.sh` | runs everything below in order; safe to restart |
| `prepare.py` | held-out split, generation lists, real reference crops |
| `train.py` | `--stage base` or `--stage finetune` |
| `generate.py` | images from one saved model for a list of masks and seeds |
| `evaluate.py` | LPIPS, paired comparisons, KID/FID, the v1 replication; writes `metrics/summary.md` |
| `config.py`, `data.py`, `model.py`, `utils.py` | settings and code copied from the v1 notebook |
| `smoke_test.py` | laptop test of the whole pipeline |
| `setup_ec2.sh`, `requirements.txt` | environment for the GPU machine |

## Running it on EC2

**Expected cost:** about 10 GPU-hours on a `g5.xlarge` (roughly $10–12 on-demand),
if training runs near 1 step per second. The training log prints the real speed
and an ETA every 100 steps, so you will know within minutes.

### 0. One-time AWS preparation
1. In the AWS console, open **Service Quotas → Amazon EC2 → Running On-Demand G and VT instances**
   and request **4** vCPUs (us-east-1). Approval can take a day or more.
2. Create a key pair (EC2 → Key pairs) and save the `.pem` file, e.g. `~/.ssh/rerun-key.pem`,
   then run `chmod 400 ~/.ssh/rerun-key.pem`.
3. Set a billing alarm (Billing → Budgets) at, say, $30.

### 1. Launch the machine
EC2 → Launch instance:
* AMI: search for **Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**
* Instance type: **g5.xlarge**
* Key pair: the one from step 0
* Network: allow SSH from **My IP** only
* Storage: **100 GB gp3**

Note the instance's public IP address. Below it is written `IP`.

### 2. Copy the code and data (from your Mac)
```bash
cd "/Users/yusufabdulkadir/Leprosy AI Project"
rsync -avz -e "ssh -i ~/.ssh/rerun-key.pem" rerun ubuntu@IP:~/
rsync -avz -e "ssh -i ~/.ssh/rerun-key.pem" --exclude ".ipynb_checkpoints" final_dataset ubuntu@IP:~/data/
```
Only these two folders go to AWS. Your `.env` file with the OpenAI key stays on your Mac.

### 3. Set up (on the machine)
```bash
ssh -i ~/.ssh/rerun-key.pem ubuntu@IP
bash ~/rerun/setup_ec2.sh
```
This installs the packages and downloads Stable Diffusion, LPIPS and Inception weights (about 5 GB).

### 4. Run
```bash
tmux new -s rerun
source ~/venv-rerun/bin/activate
SHUTDOWN_WHEN_DONE=1 bash ~/rerun/run_all.sh
```
`tmux` keeps the run going if your laptop disconnects. Detach with `Ctrl-b` then `d`;
reattach later with `tmux attach -t rerun`. Watch progress with
`tail -f ~/runs/rerun1/logs/train_base.log`.

With `SHUTDOWN_WHEN_DONE=1` the machine **stops** itself when the run ends, so it
stops billing for GPU time. Its disk is kept (about $8/month) so you can download results.

If training fails with a CUDA out-of-memory error, start it again with
`TRAIN_ARGS=--gradient_checkpointing SHUTDOWN_WHEN_DONE=1 bash ~/rerun/run_all.sh`.
This lowers memory use and does not change the results.

### 5. If the run is interrupted
Start the instance again, reconnect, and run the same command from step 4. Finished
steps are skipped and training resumes from its last checkpoint.

### 6. Download everything to your Mac
Start the instance if it is stopped (its IP may change), then on your Mac:
```bash
mkdir -p "/Users/yusufabdulkadir/Leprosy AI Project/runs"
cd "/Users/yusufabdulkadir/Leprosy AI Project/runs"
rsync -avz -e "ssh -i ~/.ssh/rerun-key.pem" ubuntu@IP:~/runs/rerun1 ./
```
That copies the metrics, both trained models (about 7 GB) and all generated images.
`rerun1/metrics/summary.md` holds the results; `rerun1/results.tar.gz` is the same
material without the models and images.

### 7. Terminate the instance
Once the download has finished and you can open `summary.md`, terminate the instance
(EC2 → Instances → Terminate). This deletes its disk and ends all charges.

## What the results contain

`metrics/summary.md`:
* **Held-out set-level LPIPS**: real–real, base–base, fine-tuned–fine-tuned, real–base,
  real–fine-tuned, fine-tuned–base, with 95% cluster-bootstrap intervals.
* **Paired LPIPS**: each held-out mask goes through both models with the same seed. This is set
  against the distance between two seeds of one model, so it shows whether fine-tuning changes an
  image more or less than resampling does. A third set runs the base model with the leprosy prompt,
  to separate the effect of the weights from the effect of the prompt.
* **KID and FID** of each generated set against the real held-out images. KID is the main number;
  FID is biased with about 136 real images.
* **Replication**: the v1 protocol rerun side by side with the v1 numbers (0.672, 0.662, 0.663, 0.716).

`metrics/heldout_examples.png` shows mask, real image, base output and fine-tuned outputs side by side.
`metrics/pairs/*.csv` holds every individual score.

## Testing changes on a laptop
```bash
cd rerun
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python smoke_test.py
```
It takes about a minute and downloads nothing. Its numbers are meaningless (the networks are random);
it checks that every step runs and that the outputs are wired together correctly.

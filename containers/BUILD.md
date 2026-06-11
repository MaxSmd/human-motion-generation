# Container build (enroot, primary path)

You build the container **once** on the cluster and reuse it from every sbatch.
Mirrors the cluster's official Quickstart (Enroot section).

## Prerequisites

- VPN to TUM connected; `ssh head` works.
- `containers/requirements.txt` is committed in this repo and reachable from
  inside an interactive job (e.g. via `/mnt/home/<user>/rmg/`).

## Build

```bash
# 1. Open an interactive job on a 24g node.
ssh head
srun --partition=24g --qos=students_normal --gres=gpu:1 --pty bash -l

# 2. Pull the upstream PyTorch image as an enroot squashfs.
enroot import -o /tmp/base.sqsh 'docker://pytorch/pytorch:2.9.0-cuda13.0-cudnn9-devel'

# 3. Create a writable container from it.
enroot create --force --name rmg /tmp/base.sqsh

# 4. Enter it (rw, with cluster bind mounts).
enroot start --root --rw --mount /mnt:mnt --mount /tmp:tmp rmg
```

Inside the container:

```bash
apt-get update && apt-get install -y --no-install-recommends git
pip install --no-cache-dir -r /mnt/home/<user>/rmg/containers/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
exit
```

Save the image:

```bash
enroot export --force --output ~/rmg.sqsh rmg
```

That's it. From now on every sbatch starts a fresh container from `~/rmg.sqsh`.

## Smoke test the saved image

```bash
OVERRIDES='data.subset_n=4 train.max_steps=25' sbatch slurm/rmg_train.sbatch
cat slurm/logs/smoke.out  # path is set inside the sbatch script
```

If you see `torch <ver> cuda? True` and our `import rmg` line, the container
is ready.

## Re-building (when `requirements.txt` changes)

The lazy way: open an interactive job, `enroot start --root --rw ~/rmg.sqsh`
(read the existing image), `pip install -U <pkg>`, `exit`, re-export. Or
rebuild from base.

## Alternative: podman + Dockerfile

If you prefer a non-interactive build, use the Dockerfile. From the head node
or a `data`-partition job:

```bash
podman build -t rmg containers/
enroot import -o ~/rmg.sqsh podman://rmg
```

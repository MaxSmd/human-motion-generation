"""Cluster control plane: SSH/rsync/sbatch broker over the user's own SSH config.

See `cluster-plan.md`. The backend runs on the host (so it inherits the VPN
route + `~/.ssh`); connectivity is one temporary SSH ControlMaster opened for the
app session. Nothing heavy runs locally — the model and the animated render live
on the GPU node; the laptop only pulls small artifacts and serves them.
"""

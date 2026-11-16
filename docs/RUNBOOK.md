# RUNBOOK — WSL2 + RTX 4060 Ti 8 GB

## 0. Prerequisiti (una tantum)
1. **Driver NVIDIA solo su Windows**, versione ≥ R580 (le wheel Warp/CUDA 13 la richiedono; con
   `--extra cu128` basta un driver recente CUDA 12.8). **Mai** installare un driver Linux dentro WSL2.
2. In PowerShell: `wsl --update`, poi `wsl -l -v` (la distro Ubuntu deve essere VERSION 2).
3. Limita la RAM di WSL2 (host 16 GB): crea `C:\Users\<tu>\.wslconfig`
   ```
   [wsl2]
   memory=12GB
   swap=8GB
   ```
   poi `wsl --shutdown`.
4. In Ubuntu: `nvidia-smi` deve mostrare la 4060 Ti. Se non la vede, ci si ferma qui (driver Windows).
5. Installa `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`). Il `mujoco 3.8.0` di sistema
   non interferisce: il progetto usa solo il suo venv (`.venv`, mujoco 3.11 fissato da mjlab).

## 1. Setup
```bash
cd ~                       # filesystem Linux, NON /mnt/c (I/O molto più lento)
git clone --recurse-submodules <repo> x2-locomotion-mjlab && cd x2-locomotion-mjlab
uv sync --extra cu128 --extra dev --extra deploy   # genera uv.lock: committalo (non incluso nell'archivio)
uv run python -c "import torch, warp as wp; print(torch.cuda.is_available()); wp.init()"
uv run pytest -q -m "not slow"
```

## 2. Smoke GPU prima di ogni run lunga (NON ancora verificato su questa macchina)
```bash
uv run train X2-Velocity-Flat --env.scene.num-envs 512 --agent.max-iterations 20
nvidia-smi                 # annota VRAM usata e step/s
```
Poi sali: 1024 → 2048 → 4096 finché la VRAM resta < ~7 GB. Se i contatti saturano (errori
`nconmax`/`njmax`), aumenta i buffer in `env_cfgs.py` (sono stime).

## 3. Run
```bash
# Fase 1
uv run train X2-Velocity-Flat --env.scene.num-envs 2048 --agent.max-iterations 3000
# Fase 2 (fine-tune, solo pesi, lr 5e-4)
uv run train X2-Velocity-GentleBlind-Finetune --env.scene.num-envs 2048 \
    --agent.resume True --agent.load-run '<cartella run fase 1>' --agent.load-checkpoint 'model_3000.pt'
# Opzionale: da zero
uv run train X2-Velocity-GentleBlind-Scratch --env.scene.num-envs 2048
```
Log in `logs/rsl_rl/x2_velocity/<run>/` (TensorBoard: `uv run tensorboard --logdir logs`).
Ogni checkpoint `model_N.pt` ha il suo `.onnx` accanto.

## 4. Valutazione, video, ONNX
```bash
uv run python scripts/evaluate.py --terrain gentle --label base_zero_shot --checkpoint <fase1.pt>
uv run python scripts/evaluate.py --terrain gentle --label finetuned      --checkpoint <fase2.pt>
uv run python scripts/make_table.py results/eval.csv > results/table.md
uv run python scripts/onnx_parity.py --terrain gentle --checkpoint <fase2.pt>
uv run play X2-Velocity-GentleBlind-Finetune --agent trained --checkpoint-file <fase2.pt> --video True
```
Senza display (WSLg assente) usa `--video True` o `--viewer viser`.

## 5. Regole di igiene
* Una sola modifica per run; cambia `--agent.run-name` ogni volta.
* Copia i checkpoint buoni in `models/` e annota il commit git (`evaluate.py` lo scrive nel CSV).
* Se la run è lunga: tienila in `tmux`; il PC non deve andare in sospensione.
* Se WSL2/GPU dà problemi: fallback su Colab (stesso `uv sync`, GPU T4/L4), poi evaluate in locale.

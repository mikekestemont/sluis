# sluis
Computational writer identification in the Sluis fragment of the Spiegel historiael

## Environment

Conda env `sluis`, Python 3.10.18. Tracked files: `environment.yml` (conda) and
`requirements.txt` (pip). After a successful create, `requirements.lock` is a
full `pip freeze` of that env.

```bash
conda env create -f environment.yml
conda activate sluis
python -m ipykernel install --user --name sluis --display-name "Python (sluis)"
```

Recreate from scratch:

```bash
conda env remove -n sluis
conda env create -f environment.yml
```

Kraken BLLA (text zones) needs this env, not the system Python:

```bash
conda activate sluis
python code/05_blla_zones.py --device mps --qc-every 0
```

Mole stays in its own repo and environment (`mole` / `mole[detect]`). Do not
install mole into `sluis`.


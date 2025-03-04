# SingleBeadSAXS

## Install

```
conda create -n singlebeadsaxs python=3.8 -y
conda activate singlebeadsaxs
conda install -c defaults -c conda-forge -c https://conda.rosettacommons.org pyrosetta
```
Get Dunbrack rotamer library : 
```
cd /path/to/singlebeadsaxs
wget https://dl.fbaipublicfiles.com/protein-ebm/dunbrack_rotamer.tar.gz
tar xzvf dunbrack_rotamer.tar.gz
```


## Run
```
mkdir outputs
python ./single_bead_formfactor.py outputs/single_beads
```
or (martini)
```
python ./single_bead_formfactor.py outputs/martini_beads --martini_beads
```

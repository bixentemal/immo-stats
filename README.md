## DVF stats map

This tool requires to first download data from DVF :
https://www.data.gouv.fr/fr/datasets/demandes-de-valeurs-foncieres/

### How to

First convert the DVF file to a csv containing only usefull data :

```
python prepare_dataset.py valeurfoncieres/ LANCIEUX SAINT-BRIAC-SUR-MER SAINT-CAST-LE-GUILDO
```

Then launch jupyter
```
jupyter notebook
```
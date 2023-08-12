import json
import os

with open("bnd_lista.json", "r", encoding="utf-8") as arquivo:
    bnd_lista = json.load(arquivo)

bnd_lista = bnd_lista[:-1000]

novo_indice = 0
for indice in bnd_lista:
    novo_indice += 1;
    if indice < 3616:
        os.system(f"mv databases/CRD/COVID/COVID-{indice}.png databases/CRD/BND/BND-{novo_indice}.png")
    else:
        os.system(f"mv databases/CRD/Normal/Normal-{indice-3615}.png databases/CRD/BND/BND-{novo_indice}.png")

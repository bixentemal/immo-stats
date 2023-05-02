import argparse
from typing import List
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from numpy import NaN
from pandas import DataFrame
import os
import datetime as dt


def getlatlong(numvoie, typevoie, voie, postalcode, commune) :
    if typevoie == "BD":
        typevoie == "BOULEVARD"
    if typevoie == "IMP":
        typevoie = "IMPASSE"
    if typevoie == "CHE":
        typevoie = "CHEMIN"
    q = urlencode({'q': '%s %s %s %s %s' % (str(int(numvoie)), str(typevoie), str(voie), str(int(postalcode)), str(commune)),
                   'format': 'json', 'polygon': '1', 'addressdetails': "1"})
    url = "https://nominatim.openstreetmap.org/search?"+q
    x = requests.get(url)
    d = x.json()
    print("%s %s"%(url, str(x.status_code)))
    if len(d) > 0:
        return d[0]['lat'], d[0]['lon']
    else:
        return NaN,NaN

def get_dataset_list(dir_path: str) -> List[str]:
    res = []

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(dir_path + "/" + path)
    return res

def transform(df:DataFrame) -> DataFrame:
    pass

def main(dataset_file_pathes: List[str], filter=None, outfile = "./converted.csv"):
    df = None
    for i in dataset_file_pathes:
        tmp = pd.read_csv(i, sep="|", usecols=["Commune",
                                               "Date mutation",
                                               "Nature mutation", "Valeur fonciere",
                                                "No voie",
                                                "Type de voie",
                                                "Voie",
                                                "Code postal",
                                                "Code type local", #1: maison; 2: appartement; 3 : dépendance (isolée) ; 4 : local industriel et commercial ou assimilés
                                                "Type local",
                                                "Surface reelle bati",
                                                "Nombre pieces principales",
                                                "Nature culture",
                                                "Surface terrain",
                                                "No disposition",
                                                "Nombre de lots",
                                                "Identifiant local",
                                                "Identifiant de document",
                                                "Reference document"])
        if df is not None:
            df = pd.concat([tmp, df])
        else:
            df = tmp
    if filter:
        df = df[df["Commune"].isin(filter)]
    # clean
    df = df[df["Nombre pieces principales"] > 0]
    df = df.dropna(subset=["Surface reelle bati"])
    grpby = df.groupby(['Date mutation', 'Nature mutation', 'Valeur fonciere', 'No voie', 'Type de voie', "Voie", 'Code postal', "Commune"])
    a = grpby.apply(lambda x: np.sum(a=x['Surface terrain'])).to_frame(name='Surface terrain')
    b = grpby.apply(lambda x: np.max(a=x['Surface reelle bati'])).to_frame(name='Surface reelle bati')
    c = grpby.apply(lambda x: np.max(a=x['Code type local'])).to_frame(name='Code type local')
    df = a.join(b).join(c)
    df = df.reset_index()
    df["PrixM2Terrain"] = df["Valeur fonciere"].str[:-3].astype(float) / df["Surface terrain"]
    df["PrixM2Bati"] = df["Valeur fonciere"].str[:-3].astype(float) / df["Surface reelle bati"]
    df2 = pd.DataFrame(
        map(getlatlong,
            df['No voie'], df['Type de voie'], df['Voie'], df['Code postal'], df['Commune']),
    columns=['lat', 'lon'])
    df2 = df2.apply(pd.to_numeric)
    df = df.join(df2)
    df = df.dropna()
    df['datetime'] = pd.to_datetime(df['Date mutation'], format='mixed')
    df['now'] = dt.datetime.now()
    df['days_since'] = (df['now'] - df['datetime']).dt.days
    df['monthes_since'] = (df['days_since'] / 30).astype(int)
    # apply inflation coeff
    df["ajusted_coef"] = 1.01 ** df['monthes_since']
    df['price'] = pd.to_numeric(df['Valeur fonciere'].str[:-3].astype(float))
    df["ajusted_price"] = df["price"] * df["ajusted_coef"]
    df = df.rename(columns={"price": "price", "Surface terrain": "terrain",
                            "Surface reelle bati": "building", "Nature mutation": "mutation",
                            "Code type local": "typelocal", "Date mutation": "date",
                            'No voie' : 'roadnum',
                            'Type de voie' : 'roadtype',
                            "Voie" : 'road',
                            'Code postal' : 'zipcode',
                            "Commune": 'city'})
    df = df[["price", "ajusted_price", "terrain", "building", "mutation", "typelocal", "date", 'roadnum', 'road','zipcode','city', 'lat', 'lon']]
    #print(df.to_string())
    df.to_csv(outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        prog="mamba",
        description="Extract DVB data to converted CSV",
        epilog=""
    )

    # first positional argument : run, describe
    parser.add_argument('dvf_path',
                        type=str, nargs=1,
                        help="the path to dir containing DVF extracts.")

    parser.add_argument('--out', dest='out_file', type=str, default="converted.csv", help='converted csv files')

    parser.add_argument('cities', metavar='city', type=str, nargs='+',
                        help='the list of cityies to consider for extract.')

    args = parser.parse_args()

    dvf_path = args.dvf_path
    cities_list = args.cities
    out = args.out_file

    print("Cities = %s"%(str(cities_list)))
    print("DVF path = %s"%(dvf_path[0]))
    print("Out csv = %s" % (out))

    main(get_dataset_list(dvf_path[0]),
         cities_list, out)


import os
import pandas as pd


sciezka_katalogu = "//home//patryk//danne"

df = pd.read_csv('data_cale.csv')


for rok in os.listdir(sciezka_katalogu):

    rok_sciezka = os.path.join(sciezka_katalogu, rok)

    if os.path.isdir(rok_sciezka):
        for plik in os.listdir(rok_sciezka):
            sciezka_pliku = os.path.join(rok_sciezka, plik)

            dane_dnia = pd.read_csv(sciezka_pliku)
            df = pd.concat([df, dane_dnia], ignore_index=True)
            print(f"{plik}")
	
				

print(df.info())
print(df.describe())
print(df)

df.to_csv(f"data_cale.csv", index=False)



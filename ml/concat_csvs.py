import os
import pandas as pd

# Ścieżka do katalogu "dane"
sciezka_katalogu = "//home//patryk//danne"

# Inicjalizacja pustego DataFrame
df = pd.DataFrame()

# Iteracja przez katalogi w "dane"
for rok in os.listdir(sciezka_katalogu):
    rok_sciezka = os.path.join(sciezka_katalogu, rok)

    # Sprawdzenie, czy to jest katalog
    if os.path.isdir(rok_sciezka):
        # Iteracja przez pliki CSV w danym roku
        for plik in os.listdir(rok_sciezka):
            sciezka_pliku = os.path.join(rok_sciezka, plik)

            # Odczytanie danych z pliku CSV i dodanie ich do DataFrame
            dane_dnia = pd.read_csv(sciezka_pliku)
            df = pd.concat([df, dane_dnia], ignore_index=True)
            print(f"{plik}")
	
    # tylko 2018
    df.to_csv(f"data_{rok}.csv", index=False)
    break
	
				

# Wyświetlenie ostatecznego DataFrame
print(df.info())
print(df.describe())
print(df)




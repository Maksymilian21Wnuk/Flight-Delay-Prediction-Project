from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.float_format', lambda x: '%.5f' % x)

def mapping(val, whattomap):
    # Function to map values using a predefined mapping from a JSON file
    with open('mappings.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        data_dict = data.get(whattomap)
        resp_dict = data_dict.get(str(val))
    return resp_dict

df = pd.read_csv("data_2018.csv", usecols=lambda c: c != 'CONDITIONS')





def delays_barplot():
    delay_counts = df['DELAY'].value_counts()
    ax = sns.barplot(x=delay_counts.index, y=delay_counts, hue=delay_counts.index, palette="Set2", legend=False)
    ax.set_xticks((0, 1))
    ax.set_xticklabels(["No", "Yes"])
    plt.show()


# JESZCZE MAPPOWANIE
def airlines_barplot():
	ax = sns.countplot(x='OP_CARRIER', hue='DELAY', data=df, palette='Set2')
	plt.show()


# JESZCZE MAPPOWANIE I NOWA KOLUMNA Z DNIAMI TYGODNIA ??
def dayoftheweek_braplot():
	ax = sns.countplot(x='FL_DAY', hue='DELAY', data=df, palette='Set2')
	plt.show()


# Normalized Mutual Information
# target='ARR_DELAY'
# POMIESZNAE DF Z DF2 BO DANE NIE OBROBIONE
def NMI(cols, target=None):
	# cols2 = cols + [target]
	# df2 = df[cols2]
	# df2 = df2.dropna() 
	if target:
		nmi_values = []
		for col in cols:
			nmi_value = normalized_mutual_info_score(df[col], df[target])
			nmi_values.append((col, nmi_value))
		nmi_values_sorted = sorted(nmi_values, key=lambda x: x[1], reverse=True)
		print(f"{target}")
		for col, nmi_value in nmi_values_sorted:
			print(f"{col:<15}{round(nmi_value, 3)}")
	else:
		cols.append(target)
		nmi_matrix = pd.DataFrame(index=cols, columns=cols)
		for col1 in cols:
			for col2 in cols:
				nmi_value = normalized_mutual_info_score(df[col1], df[col2])
				nmi_matrix.loc[col1, col2] = nmi_value
		heatmap_nmi = sns.heatmap(nmi_matrix.astype(float), annot=True, cmap="coolwarm", vmin=0, vmax=1)
		plt.title("Normalized Mutual Information (NMI) Heatmap")
		plt.show()


# Linear Correlation Heapmap
# target='ARR_DELAY'
# TYLKO NA OBROBIONYCH DNAYCH
def linear_correaltaion(cols, target=None):
	if target:
		correlations = round(df[cols].corrwith(df[target]), 3)
		print(f"{target}")
		for col, correlation in zip(cols, correlations):
			print(f"{col:<15}{correlation}")
	else:
		sns.heatmap(df[cols].corr(), vmin=-1, vmax=1, annot=False)
		plt.show()

def columns_by_variance(cols=None):
	# if cols:
		# df = df[cols]
	# df2 = df.dropna()
	# df2=df2.drop('id', axis=1)
	features = df.columns
	X = df.values
	pca = PCA()
	pca.fit(X)
	columns_by_variance = sorted(zip(pca.explained_variance_, features), reverse=True)
	variance_sum = sum([var for var, _ in columns_by_variance])
	print("Columns ordered by variance:")
	for var, col in columns_by_variance:
	        print(f"{col:<18}{round(var/variance_sum, 4)}")








# jeszcze bedize robione
# dziala na polaczonych danych w jednego csv bez nanow, stringow itp
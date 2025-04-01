# Análisis de valores únicos
print("\nUnique values in selected features:")
for feature in ['Gr Liv Area', 'SalePrice', 'Overall Qual']:
    print(f"{feature}: {df[feature].nunique()} unique values")

# Estadísticas descriptivas
print("\nDescriptive Statistics:")
print(df[['Gr Liv Area', 'SalePrice', 'Overall Qual']].describe())

# Contar los valores únicos de todas las columnas
unique_values_count = df.nunique()
print("\nUnique values count per feature:\n", unique_values_count)

# Histograma de 'Gr Liv Area'
gr_liv_area = df['Gr Liv Area'].to_numpy(dtype=float)
plt.figure(figsize=(8, 6))
sns.histplot(gr_liv_area, kde=True, bins=30)
plt.xlabel("Gr Liv Area")
plt.ylabel("Frequency")
plt.title("Distribution of Gr Liv Area")
plt.show()

# Histograma de 'Overall Qual'
sns.histplot(df['Overall Qual'], discrete=True)
plt.xlabel("Overall Quality")
plt.ylabel("Count")
plt.title("Distribution of Overall Quality")
plt.show()
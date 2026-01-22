import pandas as pd
import matplotlib.pyplot as plt
print("Mithra Ravi - 24BAD070")
bank_df = pd.read_csv(r"C:\Users\mithr\Downloads\marketing_campaign.csv", sep="\t")
print("\nHEAD:")
print(bank_df.head())
print("\nINFO:")
bank_df.info()
print("\nMISSING VALUES:")
print(bank_df.isnull().sum())
bank_df["Age"] = 2025 - bank_df["Year_Birth"]
plt.figure()
bank_df["Age"].value_counts().sort_index().plot(kind="bar")
plt.title("Age Distribution of Customers")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
# Box plot – Income
plt.figure()
plt.boxplot(bank_df["Income"].dropna())
plt.title("Income Distribution")
plt.ylabel("Income")
plt.tight_layout()
plt.show()
# Box plot – Spending (Total Spending)
bank_df["Total_Spending"] = (
    bank_df["MntWines"] +
    bank_df["MntFruits"] +
    bank_df["MntMeatProducts"] +
    bank_df["MntFishProducts"] +
    bank_df["MntSweetProducts"] +
    bank_df["MntGoldProds"]
)

plt.figure()
plt.boxplot(bank_df["Total_Spending"])
plt.title("Customer Spending Distribution")
plt.ylabel("Total Spending")
plt.tight_layout()
plt.show()

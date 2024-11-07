import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
data = """CustomerID Count Country State City Zip_Code Lat Long Latitude Longitude Gender Senior_Citizen Partner Dependents Tenure_Months Phone_Service Multiple_Lines Internet_Service Online_Security Online_Backup Device_Protection Tech_Support Streaming_TV Streaming_Movies Contract Paperless_Billing Payment_Method Monthly_Charges Total_Charges Churn_Label Churn_Value Churn_Score CLTV Churn_Reason
9237-HQITU 1 United States California Los Angeles 90005 34.059281 -118.30742 34.059281 -118.307420 Female No No Yes 2 Yes No Fiber_optic No No No No No No Month-to-month Yes Electronic_check 70.7 151.65 Yes 1 67 2701 Moved
9305-CDSKC 1 United States California Los Angeles 90006 34.048013 -118.293953 34.048013 -118.293953 Female No No Yes 8 Yes Yes Fiber_optic No No Yes No Yes Yes Month-to-month Yes Electronic_check 99.65 820.5 Yes 1 86 5372 Moved
7892-POOKP 1 United States California Los Angeles 90010 34.062125 -118.315709 34.062125 -118.315709 Female No Yes Yes 28 Yes Yes Fiber_optic No No Yes Yes Yes Yes Month-to-month Yes Electronic_check 104.8 3046.05 Yes 1 84 5003 Moved
0280-XJGEX 1 United States California Los Angeles 90015 34.039224 -118.266293 34.039224 -118.266293 Male No No Yes 49 Yes Yes Fiber_optic No Yes Yes No Yes Yes Month-to-month Yes Bank_transfer_(automatic) 103.7 5036.3 Yes 1 89 5340 Competitor_had_better_devices
6467-CHFZW 1 United States California Los Angeles 90028 34.099869 -118.326843 34.099869 -118.326843 Male No Yes Yes 47 Yes Yes Fiber_optic No Yes No No Yes Yes Month-to-month Yes Electronic_check 99.35 4749.15 Yes 1 77 5789 Competitor_had_better_devices
6047-YHPVI 1 United States California Los Angeles 90039 34.110845 -118.259595 34.110845 -118.259595 Male No No Yes 5 Yes No Fiber_optic No No No No No No Month-to-month Yes Electronic_check 69.7 316.9 Yes 1 66 2454 Competitor_offered_higher_download_speeds
5380-WJKOV 1 United States California Los Angeles 90041 34.137412 -118.207607 34.137412 -118.207607 Male No No Yes 34 Yes Yes Fiber_optic No Yes Yes No Yes Yes Month-to-month Yes Electronic_check 106.35 3549.25 Yes 1 65 2941 Competitor_offered_higher_download_speeds
8168-UQWWF 1 United States California Los Angeles 90042 34.11572 -118.192754 34.115720 -118.192754 Female No No Yes 11 Yes Yes Fiber_optic No No Yes No Yes Yes Month-to-month Yes Bank_transfer_(automatic) 97.85 1105.4 Yes 1 70 5674 Competitor_offered_more_data
7760-OYPDY 1 United States California Los Angeles 90056 33.987945 -118.370442 33.987945 -118.370442 Female No No Yes 2 Yes No Fiber_optic No No No No Yes No Month-to-month Yes Electronic_check 80.65 144.15 Yes 1 90 5586 Competitor_offered_more_data
"""
data_io = StringIO(data)
df = pd.read_csv(data_io, sep=" ")
print("DataFrame Structure:")
print(df.info())
print("\nFirst 5 Rows:")
print(df.head())
print("\nStatistical Summary:")
print(df.describe())
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Churn_Label', palette='Set2')
plt.title('Churn Distribution')
plt.xlabel('Churn Label')
plt.ylabel('Count')
plt.show()
plt.figure(figsize=(10, 6))
sns.histplot(df['Monthly_Charges'], kde=True, bins=30)
plt.title('Distribution of Monthly Charges')
plt.xlabel('Monthly Charges')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Tenure_Months', y='Monthly_Charges', hue='Churn_Label', palette='Set2')
plt.title('Tenure vs Monthly Charges')
plt.xlabel('Tenure (Months)')
plt.ylabel('Monthly Charges')
plt.show()
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

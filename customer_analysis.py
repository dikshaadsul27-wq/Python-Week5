# -----------------------------
# Day 1: Data Loading & Exploration
# -----------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
sales = pd.read_csv("sales_data.csv")
churn = pd.read_csv("customer_churn.csv")

# Explore structure
print(sales.info())
print(sales.head())
print(churn.info())
print(churn.head())

# Check missing values
print(sales.isnull().sum())
print(churn.isnull().sum())

# -----------------------------
# Day 2: Data Cleaning & Preparation
# -----------------------------
# Convert Date column to datetime
sales['Date'] = pd.to_datetime(sales['Date'])

# Ensure numeric columns
sales['Quantity'] = pd.to_numeric(sales['Quantity'])
sales['Price'] = pd.to_numeric(sales['Price'])
sales['Total_Sales'] = pd.to_numeric(sales['Total_Sales'])

# Create calculated fields
sales['Revenue'] = sales['Quantity'] * sales['Price']

# -----------------------------
# Day 3: Customer Analysis
# -----------------------------
# Top customers by revenue
top_customers = sales.groupby('Customer_ID')['Revenue'].sum().sort_values(ascending=False).head(10)
print(top_customers)

# Merge churn data with sales
merged = sales.merge(churn, left_on='Customer_ID', right_on='CustomerID', how='left')

# Customer Lifetime Value
clv = merged.groupby('Customer_ID')['Revenue'].sum().reset_index().sort_values(by='Revenue', ascending=False)
print(clv.head())

# Regional distribution
region_sales = sales.groupby('Region')['Revenue'].sum().reset_index()
print(region_sales)

# -----------------------------
# Day 4: Sales Pattern Analysis
# -----------------------------
# Monthly trends (fixed)
# Use 'ME' for month-end or 'MS' for month-start
monthly_sales = sales.resample('ME', on='Date')['Revenue'].sum()
print(monthly_sales)

# If you prefer grouping by calendar month instead of resampling:
monthly_sales_alt = sales.groupby(sales['Date'].dt.to_period('M'))['Revenue'].sum()
print(monthly_sales_alt)

# Best-selling products
product_sales = sales.groupby('Product')['Revenue'].sum().sort_values(ascending=False)
print(product_sales)

# -----------------------------
# Day 5: Advanced Analysis
# -----------------------------
# Pivot table: Revenue by Region vs Product
pivot_region_product = pd.pivot_table(sales, values='Revenue', index='Region', columns='Product', aggfunc='sum')
print(pivot_region_product)

# Retention rate
retention_rate = (churn['Churn'] == 0).mean()
print("Retention Rate:", retention_rate)

# Cross-selling: customers buying multiple product categories
cross_sell = sales.groupby('Customer_ID')['Product'].nunique().reset_index()
multi_buyers = cross_sell[cross_sell['Product'] > 1]
print(multi_buyers.head())

# -----------------------------
# Day 6: Dashboard Creation
# -----------------------------
plt.figure(figsize=(10,6))
top_customers.plot(kind='bar', title="Top 10 Customers by Revenue")
plt.show()

monthly_sales.plot(kind='line', figsize=(10,6), title="Monthly Sales Trend")
plt.show()

pivot_region_product.plot(kind='bar', stacked=True, figsize=(10,6), title="Product Sales by Region")
plt.show()

churn['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6,6), title="Customer Churn Distribution")
plt.show()

sns.heatmap(pivot_region_product, annot=True, fmt=".0f", cmap="Blues")
plt.title("Revenue Heatmap: Product vs Region")
plt.show()

# -----------------------------
# Day 7: Report & Insights
# -----------------------------
total_revenue = sales['Revenue'].sum()
total_customers = sales['Customer_ID'].nunique()
avg_order_value = sales['Revenue'].mean()
top_customer = clv.iloc[0]

print("CUSTOMER SALES ANALYSIS REPORT")
print(f"Total Revenue: ${total_revenue:,.0f}")
print(f"Total Customers: {total_customers}")
print(f"Average Order Value: ${avg_order_value:,.0f}")
print(f"Top Customer: {top_customer['Customer_ID']} - ${top_customer['Revenue']:,.0f}")

# -----------------------------
# Analysis Questions
# -----------------------------

# 1. Most valuable customers
top_customers = sales.groupby('Customer_ID')['Revenue'].sum().sort_values(ascending=False).head(10)
print("Top 10 Customers by Revenue:")
print(top_customers)

# 2. Products that sell best together
cross_sell = sales.groupby('Customer_ID')['Product'].nunique().reset_index()
multi_buyers = cross_sell[cross_sell['Product'] > 1]
print("Customers buying multiple product categories:")
print(multi_buyers.head())

# Optional: see which product combinations are common
product_combos = sales.groupby(['Customer_ID','Product']).size().unstack(fill_value=0)
common_combos = (product_combos.T @ product_combos)  # co-occurrence matrix
print("Product co-occurrence (cross-selling opportunities):")
print(common_combos)

# 3. Regions with highest sales
region_sales = sales.groupby('Region')['Revenue'].sum().sort_values(ascending=False)
print("Revenue by Region:")
print(region_sales)

# 4. Seasonal trends
monthly_sales = sales.groupby(sales['Date'].dt.to_period('M'))['Revenue'].sum()
print("Monthly Sales Trend:")
print(monthly_sales)

monthly_sales.plot(kind='line', figsize=(10,6), title="Seasonal Sales Trend")
plt.show()

# 5. Customer retention improvement
# Retention rate
retention_rate = (churn['Churn'] == 0).mean()
print("Retention Rate:", retention_rate)

# Churn by contract type
contract_churn = churn.groupby('Contract')['Churn'].mean()
print("Churn Rate by Contract Type:")
print(contract_churn)

# Churn by payment method
payment_churn = churn.groupby('PaymentMethod')['Churn'].mean()
print("Churn Rate by Payment Method:")
print(payment_churn)

# Visualize churn distribution
churn['Churn'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6,6), title="Customer Churn Distribution")
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PART 1 - LOAD THE RAW DATA
df = pd.read_csv("Sahl Delivery-assignmets.csv")

print("Original shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nUnique Values in Text Columns:")
for column in df.select_dtypes(include="object").columns:
    print("\n", column)
    print(df[column].unique())


# ============================================================
# PART 1 - CLEANING
# 1. Remove exact duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:", df.shape)


# 2. Drop a column that is more than 90% empty
# courier_notes is almost completely empty, so we remove it.
df = df.drop(columns=["courier_notes"])

print("\nShape after dropping courier_notes:", df.shape)


# 3. Convert order_value_egp from text to number
# Remove EGP and commas first.
df["order_value_egp"] = (
    df["order_value_egp"]
    .astype(str)
    .str.replace("EGP", "", regex=False)
    .str.replace(",", "", regex=False)
    .str.strip()
)

df["order_value_egp"] = pd.to_numeric(
    df["order_value_egp"],
    errors="coerce"
)


# 4. Standardise city names
city_map = {
    "Minya": "Minya",
    "El-Minya": "Minya",
    "Minya ": "Minya",
    "Menia": "Minya",
    "El Minya": "Minya",
    "minya": "Minya",
    "MINYA": "Minya",
    " Minya": "Minya",

    "New Minya": "New Minya",
    "New Minya ": "New Minya",
    "NEW MINYA": "New Minya",
    "new minya": "New Minya",
    "El-Minya El-Gedida": "New Minya",

    "Abu Qurqas": "Abu Qurqas",
    "Abu Qorqas": "Abu Qurqas",
    "ABU QURQAS": "Abu Qurqas",
    " Abu Qurqas": "Abu Qurqas",
    "abu qurqas": "Abu Qurqas"
}

df["city"] = df["city"].map(city_map)

print("\nCities after standardisation:")
print(df["city"].value_counts())


# 5. Convert order_datetime to real datetime
# The file contains two date formats.
# Format 1: month/day/year
# Format 2: day/month/year
#
# If the first number is greater than 12,
# it must be the day.

date_text = df["order_datetime"].astype(str)

first_number = pd.to_numeric(
    date_text.str.split("/").str[0],
    errors="coerce"
)

df["order_datetime"] = pd.NaT

day_first = first_number > 12

df.loc[~day_first, "order_datetime"] = pd.to_datetime(
    date_text.loc[~day_first],
    format="%m/%d/%Y %H:%M",
    errors="coerce"
)

df.loc[day_first, "order_datetime"] = pd.to_datetime(
    date_text.loc[day_first],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

print("\nDate conversion complete.")
print("Missing dates:", df["order_datetime"].isnull().sum())


# 6. Remove physically impossible values
# Actual delivery time cannot be negative.
# Keep missing values because cancelled orders do not have
# a real delivery time.
bad_actual = df["actual_minutes"] < 0

print("\nPhysically impossible actual_minutes:", bad_actual.sum())

df = df[~bad_actual].copy()


# 7. Fill missing numeric values with the MEDIAN
# customer_rating is NOT filled because Sahl specifically said
# to leave missing ratings blank.
#
# reordered_within_30d is the target, so we also do NOT fill it.
# Cancelled orders do not have a target value.

df["distance_km"] = df["distance_km"].fillna(
    df["distance_km"].median()
)

df["actual_minutes"] = df["actual_minutes"].fillna(
    df["actual_minutes"].median()
)


# ============================================================
# CHECKPOINT BEFORE PART 2
print("\n" + "=" * 60)
print("CLEANING CHECKPOINT")
print("=" * 60)

print("\nFinal shape:", df.shape)

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nRemaining missing values:")
print(df.isnull().sum())


# ============================================================
# WHO DID NOT LEAVE A RATING?
print("\n" + "=" * 60)
print("CUSTOMER RATING CHECK")
print("=" * 60)

df["rating_missing"] = df["customer_rating"].isnull()

print("\nOrders with rating:")
print((~df["rating_missing"]).sum())

print("\nOrders without rating:")
print(df["rating_missing"].sum())

print("\nReorder rate for rated orders:")
print(
    df.loc[~df["rating_missing"], "reordered_within_30d"].mean()
)

print("\nReorder rate for orders without rating:")
print(
    df.loc[df["rating_missing"], "reordered_within_30d"].mean()
)

# ============================================================
# PART 2 - INVESTIGATION
# We use Delivered orders when analysing delivery experience.
# Cancelled orders have no real delivery time or rating.
delivered = df[df["order_status"] == "Delivered"].copy()


# ============================================================
# CREATE NEW COLUMNS
# How many minutes late was the order?
# Positive = late
# Zero = exactly on time
# Negative = early

delivered["delay_minutes"] = (
    delivered["actual_minutes"]
    - delivered["promised_minutes"]
)

# True if the order was on time or early
delivered["on_time"] = delivered["delay_minutes"] <= 0


# Create delay groups
delivered["delay_group"] = pd.cut(
    delivered["delay_minutes"],
    bins=[-np.inf, 0, 10, 20, np.inf],
    labels=[
        "On time / early",
        "1-10 min late",
        "11-20 min late",
        "More than 20 min late"
    ]
)

# Extract hour from order datetime
delivered["hour"] = delivered["order_datetime"].dt.hour

# ============================================================
# QUESTION 1
# How many customers actually come back?
overall_reorder_rate = delivered["reordered_within_30d"].mean()

print("\n" + "=" * 60)
print("QUESTION 1 - REORDER RATE")
print("=" * 60)

print(
    f"Customers who reordered within 30 days: "
    f"{overall_reorder_rate:.2%}"
)

# ============================================================
# QUESTION 2
# Is Sahl keeping its delivery promise?
late_rate = (~delivered["on_time"]).mean()
average_delay = delivered["delay_minutes"].mean()

print("\n" + "=" * 60)
print("QUESTION 2 - DELIVERY PROMISE")
print("=" * 60)

print(f"Late orders: {late_rate:.2%}")
print(f"Average delay: {average_delay:.2f} minutes")
print(
    f"On-time / early orders: "
    f"{delivered['on_time'].mean():.2%}"
)


# ============================================================
# QUESTION 3
# Does delivery experience affect reordering?
delay_result = delivered.groupby(
    "delay_group",
    observed=True
)["reordered_within_30d"].agg(
    ["mean", "count"]
)

delay_result["mean"] = delay_result["mean"] * 100

print("\n" + "=" * 60)
print("QUESTION 3 - DELIVERY DELAY VS REORDER")
print("=" * 60)
print(delay_result)

# ============================================================
# QUESTION 4
# WHERE does the bad experience come from?
# We test vehicle type, restaurant and zone.
vehicle_result = delivered.groupby(
    "vehicle_type"
)["reordered_within_30d"].agg(
    ["mean", "count"]
)

vehicle_result["mean"] = vehicle_result["mean"] * 100

print("\n" + "=" * 60)
print("VEHICLE TYPE")
print("=" * 60)
print(vehicle_result.sort_values("mean"))


restaurant_result = delivered.groupby(
    "restaurant"
).agg(
    reorder_rate=("reordered_within_30d", "mean"),
    late_rate=("on_time", lambda x: 1 - x.mean()),
    average_delay=("delay_minutes", "mean"),
    orders=("reordered_within_30d", "count")
)

print("\n" + "=" * 60)
print("RESTAURANTS")
print("=" * 60)
print(
    restaurant_result.sort_values("reorder_rate").head(10)
)


zone_result = delivered.groupby(
    "zone"
).agg(
    reorder_rate=("reordered_within_30d", "mean"),
    late_rate=("on_time", lambda x: 1 - x.mean()),
    average_delay=("delay_minutes", "mean"),
    orders=("reordered_within_30d", "count")
)

print("\n" + "=" * 60)
print("ZONES")
print("=" * 60)
print(
    zone_result.sort_values("reorder_rate")
)
# ============================================================
# IMPORTANT COMBINATION:
# Vehicle + Zone

vehicle_zone = delivered.groupby(
    ["zone", "vehicle_type"]
).agg(
    reorder_rate=("reordered_within_30d", "mean"),
    late_rate=("on_time", lambda x: 1 - x.mean()),
    average_delay=("delay_minutes", "mean"),
    orders=("reordered_within_30d", "count")
).reset_index()

print("\n" + "=" * 60)
print("VEHICLE + ZONE")
print("=" * 60)

print(
    vehicle_zone.sort_values("reorder_rate").head(10)
)

# ============================================================
# QUESTION 5
# Is the problem concentrated in specific places/times?
hour_result = delivered.groupby(
    "hour"
)["reordered_within_30d"].mean()

print("\n" + "=" * 60)
print("REORDER RATE BY HOUR")
print("=" * 60)
print(hour_result)


# ============================================================
# TEST PREP TIME

delivered["prep_group"] = pd.cut(
    delivered["prep_minutes"],
    bins=[0, 10, 15, 20, np.inf],
    labels=[
        "0-10 min",
        "11-15 min",
        "16-20 min",
        "More than 20 min"
    ]
)

prep_result = delivered.groupby(
    "prep_group",
    observed=True
)["reordered_within_30d"].agg(
    ["mean", "count"]
)

print("\n" + "=" * 60)
print("RESTAURANT PREP TIME")
print("=" * 60)
print(prep_result)


# ============================================================
# QUESTION 6
# Test something that may NOT matter.
# Payment method is a useful dead-end test.

payment_result = delivered.groupby(
    "payment_method"
)["reordered_within_30d"].agg(
    ["mean", "count"]
)

print("\n" + "=" * 60)
print("DEAD END TEST - PAYMENT METHOD")
print("=" * 60)
print(payment_result)


# Discount test
delivered["discount_group"] = pd.cut(
    delivered["discount_egp"],
    bins=[-0.1, 0, 10, 30, 100, np.inf],
    labels=[
        "No discount",
        "1-10 EGP",
        "11-30 EGP",
        "31-100 EGP",
        "More than 100 EGP"
    ]
)

discount_result = delivered.groupby(
    "discount_group",
    observed=True
)["reordered_within_30d"].agg(
    ["mean", "count"]
)

print("\n" + "=" * 60)
print("DISCOUNT TEST")
print("=" * 60)
print(discount_result)
# ============================================================
# PART 3 - DASHBOARD
# One figure with at least SIX panels.

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))


# ------------------------------------------------------------
# PANEL 1
# Overall reorder rate

reorder_counts = delivered["reordered_within_30d"].value_counts()

axes[0, 0].bar(
    ["Did not reorder", "Reordered"],
    [
        reorder_counts.get(0, 0),
        reorder_counts.get(1, 0)
    ]
)

axes[0, 0].set_title(
    "1. Did Customers Come Back?"
)

axes[0, 0].set_xlabel("Customer outcome")
axes[0, 0].set_ylabel("Number of delivered orders")


# ------------------------------------------------------------
# PANEL 2
# Delay distribution


axes[0, 1].hist(
    delivered["delay_minutes"],
    bins=30
)

axes[0, 1].axvline(
    0,
    linestyle="--"
)

axes[0, 1].set_title(
    "2. How Often Do We Miss the Promise?"
)

axes[0, 1].set_xlabel("Delay (minutes)")
axes[0, 1].set_ylabel("Number of orders")


# ------------------------------------------------------------
# PANEL 3
# Delay vs reorder
delay_plot = delivered.groupby(
    "delay_group",
    observed=True
)["reordered_within_30d"].mean()

axes[0, 2].bar(
    delay_plot.index.astype(str),
    delay_plot.values
)

axes[0, 2].set_title(
    "3. Delivery Delay vs Reordering"
)

axes[0, 2].set_xlabel("Delivery delay")
axes[0, 2].set_ylabel("Reorder rate")

axes[0, 2].tick_params(axis="x", rotation=25)


# ------------------------------------------------------------
# PANEL 4
# Vehicle type

vehicle_plot = delivered.groupby(
    "vehicle_type"
)["reordered_within_30d"].mean()

axes[1, 0].bar(
    vehicle_plot.index,
    vehicle_plot.values
)

axes[1, 0].set_title(
    "4. Vehicle Type vs Reordering"
)

axes[1, 0].set_xlabel("Vehicle type")
axes[1, 0].set_ylabel("Reorder rate")


# ------------------------------------------------------------
# PANEL 5
# Zone
zone_plot = (
    delivered.groupby("zone")["reordered_within_30d"]
    .mean()
    .sort_values()
    .reset_index()
)

axes[1, 1].barh(
    zone_plot["zone"],
    zone_plot["reordered_within_30d"]
)

axes[1, 1].set_title(
    "5. Reorder Rate by Zone"
)

axes[1, 1].set_xlabel("Reorder rate")
axes[1, 1].set_ylabel("Zone")

# PANEL 6
# Restaurant


restaurant_plot = (
    delivered.groupby("restaurant")["reordered_within_30d"]
    .mean()
    .sort_values()
    .head(10)
    .reset_index()
)

axes[1, 2].barh(
    restaurant_plot["restaurant"],
    restaurant_plot["reordered_within_30d"]
)

axes[1, 2].set_title(
    "6. Lowest-Reorder Restaurants"
)

axes[1, 2].set_xlabel("Reorder rate")
axes[1, 2].set_ylabel("Restaurant")


plt.tight_layout()

plt.savefig(
    "sahl_dashboard.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
# PART 4 - FINAL RECOMMENDATION


print("\n" + "=" * 70)
print("PART 4 - RESULTS TO USE IN THE WRITTEN REPORT")
print("=" * 70)

overall_reorder_rate = delivered["reordered_within_30d"].mean()
late_rate = (~delivered["on_time"]).mean()
average_delay = delivered["delay_minutes"].mean()

delay_table = delivered.groupby(
    "delay_group",
    observed=True
)["reordered_within_30d"].mean()

on_time_reorder = delay_table.get("On time / early", np.nan)
severe_late_reorder = delay_table.get("More than 20 min late", np.nan)

vehicle_table = delivered.groupby(
    "vehicle_type"
)["reordered_within_30d"].mean().sort_values()

zone_table = delivered.groupby(
    "zone"
)["reordered_within_30d"].mean().sort_values()

restaurant_table = delivered.groupby(
    "restaurant"
)["reordered_within_30d"].mean().sort_values()

payment_table = delivered.groupby(
    "payment_method"
)["reordered_within_30d"].mean()

worst_vehicle = vehicle_table.index[0]
worst_vehicle_rate = vehicle_table.iloc[0]

worst_zone = zone_table.index[0]
worst_zone_rate = zone_table.iloc[0]

worst_restaurant = restaurant_table.index[0]
worst_restaurant_rate = restaurant_table.iloc[0]

payment_spread = payment_table.max() - payment_table.min()

print(f"Overall reorder rate: {overall_reorder_rate:.2%}")
print(f"Late delivery rate: {late_rate:.2%}")
print(f"Average delay: {average_delay:.2f} minutes")
print(f"On-time/early reorder rate: {on_time_reorder:.2%}")
print(f"More-than-20-min-late reorder rate: {severe_late_reorder:.2%}")
print(f"Worst vehicle by reorder rate: {worst_vehicle} ({worst_vehicle_rate:.2%})")
print(f"Worst zone by reorder rate: {worst_zone} ({worst_zone_rate:.2%})")
print(f"Worst restaurant by reorder rate: {worst_restaurant} ({worst_restaurant_rate:.2%})")
print(f"Payment-method reorder-rate spread: {payment_spread:.2%}")

print("""
The written recommendation should explain:
1. The main causes of poor customer retention.
2. How the 100,000 EGP should be spent.
3. At least one dead-end finding.
4. One important limitation of the observational data.
""")

# FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATA CHECK")
print("=" * 70)

print("Final dataset shape:", df.shape)
print("Duplicate rows:", df.duplicated().sum())
print("\nRemaining missing values:")
print(df.isnull().sum())

print("\nDashboard saved as: sahl_dashboard.png")
print("Assignment script finished successfully.")

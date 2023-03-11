import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from joblib import load
import numpy as np
import torch
from torch import nn
from model import Model # load model class
# hide hamburger
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# ----------------------------------Cache Resource-----------------------------------
@st.cache_resource
def load_data():
    df = pd.read_parquet("Data/cleaned_data.parquet")
    df_pickup_dropof_zones = pd.read_csv("Data/pickup_dropoff_zones.csv")
    md = Model()
    md.load_state_dict(torch.load("weights/weights1.pt"))
    scaler = load("std_scaler.bin")
    return df, df_pickup_dropof_zones, md, scaler

# -----------------------------------Set Config--------------------------------------
st.set_page_config(page_title="Sales Dashboard", page_icon=":bar_chart:", layout="wide")
# -----------------------------------Select Box--------------------------------------
add_selectbox = st.sidebar.selectbox(
    "Please choose",
    ("Data Vizualization Part", "Machine Learning Part")
)
df, df_pickup_dropof_zones, md, scaler = load_data()

if add_selectbox == "Data Vizualization Part":
            # ------------------------------------Header-----------------------------------------
    st.subheader("Please Select Boroughs")
    # -----------------------------------Top Bar-----------------------------------------

    borough_topbar = st.multiselect("(If not, whole data will be used)", ("Bronx","Brooklyn","Manhattan","Queens","Staten_Island"))
    if borough_topbar == []:
        borough_topbar = ["Bronx","Brooklyn","Manhattan","Queens","Staten_Island"]

    new_df = df[df.pickup_boro.isin(borough_topbar)]
    pie_chart_df = new_df[["vendorid", "pickup_boro"]].groupby(by=["pickup_boro"]).count()
    jointplot_df1 = new_df[["hour_interval", "fare_amount"]].groupby(by=["hour_interval"]).mean()
    jointplot_df2 = new_df[["hour_interval", "fare_amount"]].groupby(by=["hour_interval"]).count()

    payment_type_df = new_df[["payment_type_name", "passenger_count"]].groupby(by=["payment_type_name"]).count()
    payment_type_df.passenger_count = [str(i)+"%" for i in list(round(payment_type_df.passenger_count/payment_type_df.passenger_count.sum() * 100, 2))]

    passenger_df = new_df[["payment_type_name", "passenger_count"]].groupby(by=["passenger_count"]).count()
    passenger_df.payment_type_name = [str(i)+"%" for i in list(round(passenger_df.payment_type_name/passenger_df.payment_type_name.sum() * 100, 2))]

    top_pickup_df = new_df[["fare_amount", "pickup_zone"]].groupby(by=["pickup_zone"]).count().sort_values(by=["fare_amount"], ascending=False).head(10)
    top_pickup_df.fare_amount = [str(i)+"%" for i in list(round(top_pickup_df.fare_amount/new_df.shape[0] * 100, 2))]
    top_pickup_df.columns = ["% of Total Trips"]

    top_dropoff_df = new_df[["fare_amount", "drop_zone"]].groupby(by=["drop_zone"]).count().sort_values(by=["fare_amount"], ascending=False).head(10)
    top_dropoff_df.fare_amount = [str(i)+"%" for i in list(round(top_dropoff_df.fare_amount/new_df.shape[0] * 100, 2))]
    top_dropoff_df.columns = ["% of Total Trips"]

    trip_type_df = new_df[["fare_amount", "trip_type_names"]].groupby(by=["trip_type_names"]).count()

    # -----------------------------------Columns Layer 1----------------------------------------
    st.markdown("""---""")
    left_column, middle_column, right_column = st.columns(3)
    with left_column:
        st.subheader("Total Number of Observations:")
        st.subheader(f"{new_df.shape[0]}")
    with middle_column:
        st.subheader("Average Fare Amount:")
        st.subheader(f"US ${round(float(new_df.fare_amount.mean()), 2)}")
    with right_column:
        st.subheader("Average Trips Per Day:")
        st.subheader(f"{new_df.shape[0]//333}")
    st.markdown("""---""")
    # -----------------------------------Column Layer 2----------------------------------------
    col1, col2, col3 = st.columns(spec=[1,1,2])
    # ----------------------------------------------------------------------------------------
    with col1:
        st.dataframe(payment_type_df, width=400, height=300)
    with col2:
        st.dataframe(passenger_df, width=200, height=300)
    with col3:
        fig1 = plt.figure(figsize=(18,9),facecolor="lavender")
        plt.pie(list(pie_chart_df.vendorid),labels=list(pie_chart_df.index),
                    autopct=lambda pct: '{:1.1f}%'.format(pct) if pct > 5 else '',
                    shadow=True, startangle=0, colors=["skyblue", "aqua", "skyblue", "steelblue", "deepskyblue", "skyblue"],
                    textprops={'fontsize': 20})
        plt.axis('equal')
        plt.title("Number of Trips by Borough", size=40)
        st.pyplot(fig=fig1)
    st.markdown("""---""")
    # -----------------------------------Column Layer 3----------------------------------------
    col1, col2 = st.columns(spec=[1,1])
    # -----------------------------------------------------------------------------------
    with col1:
        fig1, ax1 = plt.subplots(figsize=(20,11), facecolor="lavender")
        ### barplot
        sns.barplot(ax=ax1, x=jointplot_df1.index, y=jointplot_df2.fare_amount/333, palette="dark:b")
        ax1.grid(True)
        ax1.set(xlabel=None)
        ax1.set_title("Average Number of trips & Fare Amount through Hour Intervals", size=30)
        plt.xticks(rotation=30)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.set_ylabel(ylabel="Average Taxi Trips", size=24)
        ### lineplot
        ax2 = ax1.twinx()
        sns.lineplot(ax=ax2, x=jointplot_df1.index, y=jointplot_df1.fare_amount, linewidth=3, color="firebrick")
        ax2.grid(False)
        ax2.set(xlabel=None)
        ax2.set_ylabel(ylabel="Average Fare Amount", color="red", size=24)
        ax2.tick_params(axis='y', labelcolor="red");
        st.pyplot(fig=fig1)
    #--------------------------------------------------------------------------------------------------
    with col2:
        fig2, ax = plt.subplots(figsize=(20,10),facecolor="lavender")
        # plt.figure(figsize=(8,4), facecolor="lavender")
        sns.lineplot(data=new_df[["month", "fare_amount", "vendorid_name"]].groupby(by=["vendorid_name", "month"]).mean(),
                    x="month", y="fare_amount", hue="vendorid_name", palette="mako", linewidth=4)
        ax.set_xlim((0.5, 11.5))
        ax.set_xticks(list(range(1,12)))
        ax.set_xticklabels(["January","February","March","April","May","June","July","August","September","October","November"], rotation=45)
        ax.set(xlabel=None)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.ylabel("Fare Amount ($)", size=24)
        plt.legend(prop={'size': 20})
        plt.title("Average Fare Amount per Month by Vendor", size=30)
        st.pyplot(fig=fig2)
    st.markdown("""---""")
    # ---------------------------------------- Columns Layer 4 -------------------------
    col1, col2, col3 = st.columns([3,3,4])
    with col1:
        st.subheader("Top 10 Zone by Pickup Coordinates:")
        st.dataframe(top_pickup_df)
    with col2:
        st.subheader("Top 10 Zone by Dropoff Coordinates:")
        st.dataframe(top_dropoff_df)
    st.markdown("""---""")
    with col3:
        st.subheader("Trip Type Distribution")
        my_circle = plt.Circle( (0,0), 0.7, color='lavender')
        fig1, ax1 = plt.subplots(1, 1, figsize=(8, 7), facecolor="lavender")
        _wedges, labels, percentages = ax1.pie(list(trip_type_df.fare_amount),labels=list(trip_type_df.index),
                                        autopct=lambda pct: '{:1.1f}%'.format(pct), colors=['blue','skyblue'],
                                        shadow=True, startangle=0,
                                        textprops={'fontsize': 16})
        for label, percentage in zip(labels, percentages):
            label.set_text(label.get_text() + '\n' + percentage.get_text())
            percentage.remove()

        plt.axis('equal')
        # plt.title("Trip Type Distribution", size=24)
        p = plt.gcf()
        p.gca().add_artist(my_circle)
        st.pyplot(fig=fig1)

if add_selectbox == "Machine Learning Part":
    st.subheader("Please Choose Parameters")
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        distance = st.text_input("Approximate trip distance (in miles)", "0")
    with col2:
        app_time = st.text_input("Approximate trip time (in minutes)", "0")
    with col3:
        pick_zone = st.selectbox("Pickap Zone", sorted(list(df_pickup_dropof_zones.pickup_zone.unique())))
    with col4:
        drop_zone = st.selectbox("Dropoff Zone", sorted(list(df_pickup_dropof_zones.drop_zone.unique())))
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        airport_dummy = st.selectbox("\"Yes\" if pickap location is LaGuardia or John F. Kennedy Airports",["No","Yes"])
    with col2:
        weekend_dummy = st.selectbox("\"Yes\" if trip day is weekend", ["No", "Yes"])
    with col3:
        vendor_name = st.selectbox("Vendor Name", ["LLC", "VeriFone"])
    with col4:
        payment_type = st.selectbox("Payment Type", ["Cash","Credit card", "No charge"])
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        hour_interval = st.selectbox("Hour Interval", ["02:00-05:00","05:00-08:00", "08:00-11:00",
                                       "11:00-14:00","14:00-17:00", "17:00-20:00",
                                        "20:00-23:00","23:00-02:00"])
    with col2:
        num_of_pass = st.selectbox("Number of passengers", [0, 1, 2, 3, 4, 5, 6])
    with col3:
        tip_type = st.selectbox("Trip type", ["Street-hail", "Dispatch"])
    with col4:
        month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November"])
    
    v1, v2 = float(distance), float(app_time)
    v3, v4 = df.loc[df.pickup_zone==pick_zone, "pickup_centroid_long"].unique()[0], df.loc[df.pickup_zone==pick_zone, "pickup_centroid_lat"].unique()[0]
    v5, v6 = df.loc[df.drop_zone==drop_zone, "dropoff_centroid_long"].unique()[0], df.loc[df.drop_zone==drop_zone, "dropoff_centroid_lat"].unique()[0]
    # 6 varibale
    v7, v8 = 1 if airport_dummy=="Yes" else 0, 1 if weekend_dummy=="Yes" else 0
    # 8 varibale
    v9, v10_13 = 1 if vendor_name=="VeriFone" else 0, [0,0] if payment_type=="Cash" else ([1,0] if payment_type=="Credit card" else [0,1])
    # 13 varibale
    int_mapper = {'02:00-05:00':0,'05:00-08:00': 1,'08:00-11:00': 2,'11:00-14:00': 3,'14:00-17:00': 4,'17:00-20:00': 5,'20:00-23:00': 6,'23:00-02:00': 7}
    v13_19 = [0,0,0,0,0,0,0,0]
    v13_19[int_mapper[hour_interval]] = 1
    v13_19=v13_19[1:]
    # 20 variable
    v20 = 1 if tip_type=="Street-hail" else 0
    # 21 variable
    v21_27 = [0,0,0,0,0,0,0]
    v21_27[num_of_pass] = 1
    v21_27 = v21_27[1:]
    # 27 variable
    month_mapper = {'January': 0,'February': 1,'March': 2,'April': 3,'May': 4,'June': 5,'July': 6,'August': 7,'September': 8,'October': 9,'November': 10}
    v27_37 = [0,0,0,0,0,0,0,0,0,0,0]
    v27_37[month_mapper[month]] = 1
    v27_37 = v27_37[1:]
    #  37 variable
    ls = [v1]+[v2]+[v3]+[v4]+[v5]+[v6]+[v7]+[v8]+[v9]+v10_13+v13_19+[v20]+v21_27+v27_37
    input = np.array(ls)
    scaled_input = scaler.transform(np.expand_dims(input, axis=0))
    md.eval()
    with torch.no_grad():
        pred = md(torch.Tensor(scaled_input))
    if st.button('Get Prediction'):
        st.write(f"Estimated Trip Price: ${round(float(pred),2)}")
    st.markdown("""---""")
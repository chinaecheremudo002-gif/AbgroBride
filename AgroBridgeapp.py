
# importing liberies 
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import joblib
import pickle



st.set_page_config(
    page_title="Agro Bridge",
    page_icon="🌾",   
    layout="wide"
)

import streamlit as st
import pickle

import streamlit as st
import pickle
import joblib

@st.cache_resource
def load_models():

    # -------------------
    # PICKLE MODELS
    # -------------------
    with open("agrobridge_model_data.pkl", "rb") as f:# fake data to be removed 
        model_data1 = pickle.load(f)

    with open("agrobridge_raw_features_v2.pkl", "rb") as f: # this is raw dataset
        model_data_raw1 = pickle.load(f)

    with open("agrobridge_features_v2.pkl", "rb") as f:# this is all the fetures included the one hot encoding 
        model_features1 = pickle.load(f)

    with open("agrobridge_label_encoder_v2.pkl", "rb") as f: # this is the lable encoder for the y variable 
        model_label_encoder1 = pickle.load(f)

    with open("agrobridge_price_model_v2.pkl", "rb") as f: # this is the stored model 
        model_price_model1 = pickle.load(f)

    # -------------------
    # JOBLIB MODELS
    # -------------------
    model_data2 = joblib.load("price_model_dataset_aug")

    final_pipeline = joblib.load("final_pipeline_aug.pkl")

    # -------------------
    # RETURN EVERYTHING
    # -------------------
    return (model_data1, model_data_raw1 ,model_features1,model_label_encoder1,model_price_model1,model_data2,final_pipeline)


# unpack properly
(model_data1, model_data_raw1 ,model_features1,model_label_encoder1,model_price_model1,model_data2,final_pipeline) = load_models()

All_Features=['YEAR', 'MONTH', 'STATES', 'MARKET', 'COMMODITY', 'PRICE TYPE', 'LAG_1',
       'LAG_3', 'MOVING_AVG_3', 'MIN_DIST_S', 'MEAN_DIST_S']

Month_MAP = {"January": 1,"February": 2,"March": 3,"April": 4,"May": 5,"June": 6,"July": 7,"August": 8,"September": 9,"October": 10,"November": 11,"December": 12}

st.title("🌾 Agro Bridge - Agricultural Price Intelligence System")
st.subheader("Turn agricultural data into pricing insights and better market decisions.")
with st.sidebar:
    month= st.selectbox("Enter the month name", list(Month_MAP.keys()))
    
    year= st.selectbox("Enter the year",range(2002,2028))

    state=st.selectbox("Enter the state",['Katsina','Sokoto','Borno','Kano','Jigawa','Oyo','Lagos','Kaduna','Zamfara','Abia','Gombe',
                                         'Kebbi','Yobe', 'Adamawa'])
                                         
    
    commodity=st.selectbox("Enter commodity",['Maize', 'Millet', 'Rice (imported)', 'Sorghum', 'Beans (niebe)', 'Wheat', 'Maize (white)',
                                   'Sorghum (white)', 'Rice (milled, local)', 'Gari (white)', 'Maize (yellow)', 'Rice (local)',
                                     'Sorghum (brown)', 'Yam', 'Oil (palm)', 'Cowpeas (white)', 'Groundnuts (shelled)', 
                                'Cassava meal (gari, yellow)', 'Cowpeas (brown)', 'Yam (Abuja)', 'Maize flour', 'Meat (beef)', 
                                'Meat (goat)', 'Oil (vegetable)', 'Beans (red)', 'Beans (white)', 'Groundnuts', 'Onions', 'Fish', 'Eggs', 
                                'Bananas', 'Oranges', 'Spinach', 'Watermelons', 'Cowpeas', 'Tomatoes'])
    
    market= st.selectbox("Enter the market",['Jibia (CBM)', 'Illela (CBM)', 'Mai Adoua (CBM)', 'Damassack (CBM)', 'Dawanau',
                                            'Mai Gatari (CBM)', 'Ibadan', 'Maiduguri', 'Lagos', 'Giwa', 'Kaura Namoda',
                                            'Gujungu', 'Saminaka', 'Dandume', 'Aba', 'Gombe', 'Gwandu', 'Biu', 'Potiskum',
                                            'Damaturu', 'Mubi', 'Abba Gamaram', 'Baga Road', 'Bullunkutu', 'Budum',
                                            'Custom', 'Kusawam Shanu', 'Monday', 'Tashan Bama', 'Bolori Stores',
                                            'Damaturu (Sunday Market)', 'Geidam', 'Jakusko', 'Bade (Gashua)', 'Nguru',
                                            'Yunusari', 'Yusufari', 'Bursari', 'Gujba (Buni Yadi)', 'Gulani (Tettaba)',
                                            'Madagali', 'Michika', 'Damboa', 'Monguno', 'Gwoza Central', 'Pulka',
                                            'Bama', 'Konduga', 'Dikwa Central', 'Mafa', 'Gamboru', 'Kashuwan Shanu',
                                            'Banki', 'Machina Central', 'Bayantasha', 'Shani Main Market', 'Damagum',
                                            'Rann', 'Magumeri Central', 'Ngala', 'Biriri', 'Gombi', 'Gajiram',
                                            'Nangere', 'Mubi North', 'Ngalang', 'Ngelzarma'
                                        ])
    
    Analysis_type = st.radio("🎯 Select Analysis Type", ["📈 Price Prediction Insight", "💡 Selling Advice & Best Time to Sell"],horizontal=True)

# renaming column in model_data_raw1 

model_data_raw1.rename(
    columns={"ROLLING_MEAN_3": "MOVING_MEAN_3"},
    inplace=True
)

model_data_raw1.rename(
    columns={"PRICE_STANDARDISED": "PRICE"},
    inplace=True)

# Filter historical data based on user input
df_filtered =  model_data_raw1[
    ( model_data_raw1 ["COMMODITY"] == commodity)&
    ( model_data_raw1 ["MARKET"] == market) 
   
].copy()

df_filtered2 =    model_data2[
    (   model_data2 ["COMMODITY"] == commodity)&
    (   model_data2["MARKET"] == market) 
   
].copy()

   # Create a proper date column from MONTH names and YEAR

df_filtered["DATE"] = pd.to_datetime(
    df_filtered["YEAR"].astype(str) + "-" + df_filtered["MONTH"].astype(str) + "-01"
)

# ============================
# SAFE FEATURE ENGINEERING
# ============================

data_ok = not df_filtered.empty
data_ok2 = not df_filtered2.empty

if not data_ok:
    st.warning("⚠️ No historical data found for this commodity/market combination. Using fallback values.")

# -------- Rolling Average --------
rolling_avg = (
    df_filtered["PRICE"].rolling(window=3, min_periods=1).mean().iloc[-1]
    if data_ok and len(df_filtered) > 0
    else 0
)

# -------- Lag Features --------
lag_1 = df_filtered["PRICE"].iloc[-1] if data_ok and len(df_filtered) >= 1 else 0
lag_3 = df_filtered["PRICE"].iloc[-3] if data_ok and len(df_filtered) >= 3 else lag_1

# -------- Distance Features --------
min_dist = df_filtered["MIN_DIST_S"].iloc[-1] if data_ok and len(df_filtered) >= 1 else 0
mean_dist = df_filtered["MEAN_DIST_S"].iloc[-1] if data_ok and len(df_filtered) >= 1 else 0


# ============================
# SECOND DATASET (PRICE MODEL)
# ============================

lag_12 = df_filtered2["PRICE"].iloc[-1] if data_ok2 and len(df_filtered2) >= 1 else 0
lag_32 = df_filtered2["PRICE"].iloc[-3] if data_ok2 and len(df_filtered2) >= 3 else lag_12

rolling_avg2 = (
    df_filtered2["PRICE"].rolling(window=3, min_periods=1).mean().iloc[-1]
    if data_ok2 and len(df_filtered2) > 0
    else 0
)

min_dist2 = df_filtered2["MIN_DIST_S"].iloc[-1] if data_ok2 and len(df_filtered2) >= 1 else 0
mean_dist2 = df_filtered2["MEAN_DIST_S"].iloc[-1] if data_ok2 and len(df_filtered2) >= 1 else 0

# ----------------------------
# BUILD MODEL INPUT
# ----------------------------
df_input2= pd.DataFrame({
                            "YEAR": [year],
                            "MONTH": [Month_MAP[month]],
                            "STATES": [state],
                            "MARKET": [market],
                            "COMMODITY": [commodity],
                            "LAG_1": [lag_12],
                            "LAG_3": [lag_32],
                            "MOVING_AVG_3": [rolling_avg2],
                            "MIN_DIST_S": [min_dist2],
                            "MEAN_DIST_S": [mean_dist2]
                                })
if st.button("🚀 Predict Price Signal"):

    with st.spinner("Analyzing market data... please wait"):

        # ==========================
        # PAGE 1: Prediction Insight
        # ==========================
        if  Analysis_type == "📈 Price Prediction Insight":

           
            
        
            # ----------------------------
            # INPUT PREPARATION
            # ----------------------------
            df_input = pd.DataFrame(columns=model_features1)
            df_input.loc[0] = 0

            df_input["MONTH"] = Month_MAP[month]
            df_input["YEAR"] = year
            df_input["LAG_1"] = lag_1
            df_input["LAG_3"] = lag_3
            df_input["ROLLING_MEAN_3"] = rolling_avg
            df_input["MIN_DIST_S"] = min_dist
            df_input["MEAN_DIST_S"] = mean_dist

            commodity_col = f"COMMODITY_{commodity}"
            state_col = f"STATES_{state}"

            if commodity_col in df_input.columns:
                df_input[commodity_col] = 1

            if state_col in df_input.columns:
                df_input[state_col] = 1

            # ----------------------------
            # PREDICTION
            # ----------------------------
            pred_proba = model_price_model1.predict_proba(df_input)[0] * 100
            classes = model_price_model1.classes_
            pred_index = np.argmax(pred_proba)
            label = model_label_encoder1.inverse_transform([pred_index])[0]
            confidence = pred_proba[pred_index]
            
            #PRICE PREDICTION 
            predict2= final_pipeline.predict(df_input2)[0]

            # ----------------------------
            # HEADER METRICS
            # ----------------------------
            col1, col2, col3 ,col4= st.columns(4)

            col1.metric("📊 Market Trend", label)
            col2.metric("🎯 Confidence", f"{confidence:.2f}%")
            col3.metric("⚙️ Last Price",f"{float(lag_1):.2f}")
            col4.metric( "⚙️ price prediction",f"{float(predict2):.2f}")
                            
    

            st.divider() 

            # ----------------------------
            # INSIGHT BOX
            # ----------------------------
            st.subheader("🧠 Insight")

            if label == "Rising":
                st.success("Market is trending upward. Demand pressure likely increasing.")
            elif label in ["Falling", "Failing", "Down"]:
                st.error("Market is declining. Prices likely weakening.")
            else:
                st.warning("Market is stable with low volatility.")

            st.divider()

            # ----------------------------
            # CONFIDENCE BAR
            # ----------------------------
            st.subheader("🔋 Confidence Level")

            st.progress(int(confidence))
            st.write(f"{confidence:.2f}% confidence in prediction: **{label}**")

            st.divider()

            # ----------------------------
            # RESULT TABLE
            # ----------------------------
            result_df = pd.DataFrame({
                "Trend": ['Falling', 'Rising', 'Stable'],
                "Probability (%)": pred_proba
            })

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Probability Table")
                st.dataframe(result_df, use_container_width=True)

            with col2:
                st.subheader("📉 Probability Chart")
                st.bar_chart(result_df.set_index("Trend"))

            st.subheader("Recent Price Movement")

            df_filtered_date = (df_filtered.groupby("DATE", as_index=False)["PRICE"].mean().sort_values("DATE"))
            df_filtered_date = df_filtered_date
    

            # plot clean time seriess
            st.line_chart(df_filtered_date.set_index("DATE")[["PRICE"]])

            st.divider()

            # ----------------------------
            # FOOTER
            # ----------------------------
            st.caption("Built with XGBoost + Streamlit | AgroBridge AI System")

            # ===================================
            # PAGE 2: Selling Advice
            # ===================================

        elif Analysis_type == "💡 Selling Advice & Best Time to Sell":

            st.title("💡 Selling Advice & Market Timing Assistant")

            st.markdown("""
            This tool analyzes WFP Nigeria food price trends and helps you decide:
            - 📈 When to hold
            - 📉 When to sell
            - 🏆 Best month to maximize price
            """)

            # ==============================
            # 1. CURRENT ML PREDICTION
            # ==============================
            current_price = final_pipeline.predict(df_input2)[0]

            baseline_price = lag_12

            change_pct = (current_price - baseline_price) / baseline_price

            if change_pct > 0.05:
                decision = "📈 HOLD "
            elif change_pct < -0.05:
                decision = "📉 SELL NOW "
            else:
                decision = "⚖️ WAIT (market stable)"

            monthly_stats = (
                df_filtered.groupby("MONTH")["PRICE"]
                .mean()
                .reset_index()
            )

            month_map = {
                            1: "January", 2: "February", 3: "March",
                            4: "April", 5: "May", 6: "June",
                            7: "July", 8: "August", 9: "September",
                            10: "October", 11: "November", 12: "December"
                        }

            monthly_stats["MONTH"] = monthly_stats["MONTH"].astype(int)

            # 🔥 CREATE MONTH NAME COLUMN
            monthly_stats["MONTH_NAME"] = monthly_stats["MONTH"].map(month_map)


            # ==============================
            # 3. BEST SELLING MONTH
            # ==============================
            best_row = monthly_stats.loc[monthly_stats["PRICE"].idxmax()]
            best_month = int(best_row["MONTH"])
            best_month_name = month_map[best_month]
            best_price = best_row["PRICE"]


            # ==============================
            # 4. OUTPUT SECTION
            # ==============================
            st.subheader("📊 Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("💰 Current Predicted Price", f"{current_price:.2f}")

            with col2:
                st.metric("📌 Decision", decision)

            with col3:
                st.metric("🏆 Best Month (Historical)", best_month_name)


            st.success(f"💰 Highest Average Price: {best_price:.2f}")


            # ==============================
            # 5. VISUALIZATION (FIXED)
            # ==============================
            st.subheader("📈 Historical Monthly Price Pattern")

            st.line_chart(
                monthly_stats.set_index("MONTH_NAME")["PRICE"]
            )
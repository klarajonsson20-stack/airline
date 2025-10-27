# Import necessary libraries
import streamlit as st
import pickle
import pandas as pd

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

st.title("Airline Customer Satisfaction")
st.write("Gain insights into passenger experiences and improve satisfaction through data analysis and surveys.")
# Display an image of penguins
st.image('airline.jpg', width = 400)

# Add app description section
with st.expander("What can you do with this app?"):
    st.markdown("""
    - 📝 **Fill Out a Survey:** Provide a form for users to fill out their airline satisfaction feedback.  
    - 📊 **Make Data-Driven Decisions:** Use insights to guide improvements in customer experience.  
    - 🧩 **Interactive Features:** Explore data with fully interactive charts and summaries!  
    """)

@st.cache_resource
def load_model(path="decision_tree_airline.pickle"):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_default_raw(path="airline_train_raw.csv"):
    return pd.read_csv(path)

model = load_model()
default_X_raw = load_default_raw()
feature_cols = list(default_X_raw.columns)  # raw (pre-dummy) training columns

# Create a sidebar for user inputs
st.sidebar.subheader('**Airline Customer Satisfaction Survey**')
st.sidebar.write("**Part 1: Customer Details**")
st.sidebar.write("Provide information about the customer flying")

customer_type = st.sidebar.selectbox("What type of customer is this?", ("Loyal Customer", "disloyal Customer"))

type_of_travel = st.sidebar.selectbox("Is the customer travelling for business or personal reasons?", ("Business travel", "Personal Travel"))

class_type = st.sidebar.selectbox("In which class is the customer flying?", ("Business", "Eco", "Eco Plus"))

age = st.sidebar.number_input("How old is the customer?", 1, 100, 25)

st.sidebar.write("**Part 2: Flight details**")
st.sidebar.write("Provide details about the customers flight details")

flight_distance = st.sidebar.number_input("How far is the customer flying in miles?", 1, 20000, 500)


departure_delay = st.sidebar.number_input("How many minutes was the customer's departure flight delayed? (Enter 0 if no delay)", 0, 500, 0)  

arrival_delay = st.sidebar.number_input("How many minutes was the customer's flight delayed upon arrival? (Enter 0 if no delay)", 0, 500, 0)

st.sidebar.write("**Part 3: Customer experience**")
st.sidebar.write("Provide details about the customer's flight experience and satisfaction")

seat_comfort = st.sidebar.radio("How comfortable was the seat? (1-5 stars)", (1, 2, 3, 4, 5), horizontal=True)

departure_time = st.sidebar.radio("Was the departure/arrival time convenient for the customer?", [1, 2, 3, 4, 5], horizontal=True)

food_drink = st.sidebar.radio("How would the customer rate the food and drink?", [1, 2, 3, 4, 5], horizontal=True)

gate_location = st.sidebar.radio("How would the customer rate the gate location?", [1, 2, 3, 4, 5], horizontal=True)

wifi_service = st.sidebar.radio("How would the customer rate the inflight wifi service?", [1, 2, 3, 4, 5], horizontal=True)

entertainment = st.sidebar.radio("How would the customer rate the inflight entertainment?", [1, 2, 3, 4, 5], horizontal=True)

online_support = st.sidebar.radio("How would the customer rate online support?", [1, 2, 3, 4, 5], horizontal=True)

online_booking = st.sidebar.radio("How easy was online booking for the customer?", [1, 2, 3, 4, 5], horizontal=True)

onboard_service = st.sidebar.radio("How would the customer rate the onboard service?", [1, 2, 3, 4, 5], horizontal=True)

legroom_service = st.sidebar.radio("How would the customer rate the leg room service?", [1, 2, 3, 4, 5], horizontal=True)

baggage_handling = st.sidebar.radio("How would the customer rate baggage handling?", [1, 2, 3, 4, 5], horizontal=True)

checkin_service = st.sidebar.radio("How would the customer rate the check-in service?", [1, 2, 3, 4, 5], horizontal=True)

cleanliness = st.sidebar.radio("How would the customer rate cleanliness?", [1, 2, 3, 4, 5], horizontal=True)

online_boarding = st.sidebar.radio("How would the customer rate online boarding?", [1, 2, 3, 4, 5], horizontal=True)


# Create a button to submit the survey
if st.sidebar.button("Predict"):
    # 1) Create a single-row DataFrame matching training columns exactly
    user_row = pd.DataFrame([{
        "customer_type": customer_type,
        "age": age,
        "type_of_travel": type_of_travel,
        "class": class_type,
        "flight_distance": flight_distance,
        "seat_comfort": seat_comfort,
        "departure_arrival_time_convenient": departure_time,
        "food_and_drink": food_drink,
        "gate_location": gate_location,
        "inflight_wifi_service": wifi_service,
        "inflight_entertainment": entertainment,
        "online_support": online_support,
        "ease_of_online_booking": online_booking,
        "on-board_service": onboard_service,
        "leg_room_service": legroom_service,
        "baggage_handling": baggage_handling,
        "checkin_service": checkin_service,
        "cleanliness": cleanliness,
        "online_boarding": online_boarding,
        "departure_delay_in_minutes": departure_delay,
        "arrival_delay_in_minutes": arrival_delay
    }])[feature_cols]

    # 2) Combine with raw training data for consistent encoding
    combo = pd.concat([default_X_raw, user_row], ignore_index=True)

    # 3) Encode on the combined DataFrame
    encoded = pd.get_dummies(
        combo,
        columns=["customer_type", "type_of_travel", "class"],
        drop_first=False
    )

    # 4) Extract encoded user row and align with model’s expected columns
    user_X = encoded.tail(1).reindex(columns=model.feature_names_in_, fill_value=0)

    # 5) predict + probability 
    pred = model.predict(user_X)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(user_X)[0]
        # if classes are ['dissatisfied','satisfied'] this shows max confidence
        confidence = float(proba.max()) * 100.0
    else:
        confidence = None

    # 6) headline result card 
    label_color = "#d9534f" if str(pred).lower().startswith("diss") else "#28a745"
    st.markdown(
        f"""
        <div style="border:1px solid #e3e6ea;border-radius:8px;padding:22px;margin:8px 0;">
            <h3 style="margin:0 0 6px 0;">Prediction Result</h3>
            <div style="font-size:20px;">Your predicted satisfaction level is
                <b style="color:{label_color};">{pred}</b>.
            </div>
            <div style="opacity:.8;margin-top:6px;">
                {"With a confidence of {:.2f}%.".format(confidence) if confidence is not None else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Customer Demographic Analysis")

    # Compute normalized percentages once
    demo_perc = {
        "customer_type": default_X_raw["customer_type"].value_counts(normalize=True) * 100,
        "type_of_travel": default_X_raw["type_of_travel"].value_counts(normalize=True) * 100,
        "class": default_X_raw["class"].value_counts(normalize=True) * 100,
        "age_group": pd.cut(default_X_raw["age"], bins=[0,20,30,40,50,60,70,120],
                            labels=["<20","20–29","30–39","40–49","50–59","60–69","70+"]
                            ).value_counts(normalize=True) * 100
    }

    with st.expander("Customer Type Comparison"):
        st.write(f"**Customer Type:** Your selection `{customer_type}`")
        st.write(f"Percentage of fliers with this selection: **{demo_perc['customer_type'].get(customer_type, 0):.2f}%**")

    with st.expander("Type of Travel Comparison"):
        st.write(f"**Type of Travel:** Your selection `{type_of_travel}`")
        st.write(f"Percentage of fliers with this selection: **{demo_perc['type_of_travel'].get(type_of_travel, 0):.2f}%**")

    with st.expander("Flight Class Comparison"):
        st.write(f"**Flight Class:** Your selection: `{class_type}`")
        st.write(f"Percentage of fliers with this selection: **{demo_perc['class'].get(class_type, 0):.2f}%**")

    with st.expander("Age Group Comparison", expanded=False):
        # Define age groups
        bins = [0, 18, 30, 40, 50, 60, 70, 120]
        labels = ["<18", "18–30", "31–40", "41–50", "51–60", "61–70", "70+"]

        # Find user’s group
        user_group = pd.cut(pd.Series([age]), bins=bins, labels=labels).iloc[0]

        # Compute percentage of that group in data
        age_groups = pd.cut(default_X_raw["age"], bins=bins, labels=labels)
        pct = (age_groups == user_group).mean() * 100

        # Display
        st.write(f"""
        **Age Group:** Your selection: `{age}`  
        Your selected age group: `{user_group}`  
        Percentage of our fliers in this age group: **{pct:.2f}%**
        """)


else:
    st.info("ℹ️ Please fill out the survey form in the sidebar and click **Predict** to see the satisfaction prediction.")
    

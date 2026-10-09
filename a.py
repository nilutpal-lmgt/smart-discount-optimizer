import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Smart Product Discount Optimizer",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Smart Product Discount Optimizer")
st.write("A Machine Learning project using a Decision Tree Classifier.")

data = {
    "Cart_Value_INR": [
        500, 1000, 1200, 1400, 1499,
        1500, 1800, 2000, 2500, 3000,
        4000, 5000, 7500, 10000
    ],
    "Is_VIP": [
        0, 0, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1
    ],
    "Action": [
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1
    ]
}

df = pd.DataFrame(data)

X = df[["Cart_Value_INR", "Is_VIP"]]
y = df["Action"]

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X, y)

st.divider()

st.subheader("📊 Historical Customer Data")
st.dataframe(df, use_container_width=True)

st.caption("Action: 1 = Discount, 0 = Standard Pricing")

st.divider()

st.subheader("🛍️ Simulate New Customer Checkout")

cart_input = st.number_input(
    "Enter Total Cart Value (₹):",
    min_value=100,
    max_value=50000,
    value=1500,
    step=100
)

vip_input = st.checkbox("VIP / Premium Member")

vip_binary = int(vip_input)

if st.button("Optimize Price Strategy", type="primary"):

    customer_features = pd.DataFrame(
        [[cart_input, vip_binary]],
        columns=["Cart_Value_INR", "Is_VIP"]
    )

    prediction = int(model.predict(customer_features)[0])

    st.subheader("Decision Result")

    if prediction == 1:
        discount_amount = cart_input * 0.20
        final_price = cart_input - discount_amount

        st.success("🎉 Special Discount Approved!")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Original Price", f"₹{cart_input:,.2f}")

        with col2:
            st.metric("Discount (20%)", f"₹{discount_amount:,.2f}")

        st.metric("Final Price", f"₹{final_price:,.2f}")

        st.info("A 20% discount has been recommended for this customer.")

    else:
        st.info("💳 Standard Pricing: No discount approved.")

        st.metric("Final Price", f"₹{cart_input:,.2f}")

st.divider()

st.caption("College Project | Python • Streamlit • Pandas • Scikit-learn")

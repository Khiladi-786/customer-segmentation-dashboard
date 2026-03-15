def recommend(cluster):

    strategies = {
        0: "VIP Customers → Offer premium membership",
        1: "Loyal Customers → Loyalty rewards",
        2: "Potential Customers → Personalized offers",
        3: "Low Engagement → Discount campaigns",
        4: "At Risk Customers → Retention marketing"
    }

    return strategies.get(cluster, "General marketing strategy")
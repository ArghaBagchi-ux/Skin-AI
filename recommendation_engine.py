def get_recommendations(
    acne,
    skin_type
):

    recommendations = []

    if acne > 70:
        recommendations.extend([
            "Salicylic Acid Cleanser",
            "Niacinamide Serum"
        ])

    if skin_type == "Oily":
        recommendations.extend([
            "Oil-Free Moisturizer",
            "Gel Sunscreen"
        ])

    if skin_type == "Dry":
        recommendations.extend([
            "Ceramide Moisturizer",
            "Hydrating Cleanser"
        ])

    return recommendations
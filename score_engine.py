def calculate_skin_score(
    acne
    #pigmentation,
    #dark_circles
):
    score = 100

    score -= acne * 0.3
    #score -= pigmentation * 0.2
    #score -= dark_circles * 0.1

    score = max(0, score)

    return round(score, 2)
def generate_recommendation(weather, labor, material_delay, complexity, predicted_delay):
    recommendations = []

    if predicted_delay > 15:
        recommendations.append("High delay risk: Re-sequence critical construction tasks.")
    
    if labor < 50:
        recommendations.append("Low labor availability: Increase workforce allocation or use shift scheduling.")
    
    if material_delay == 1:
        recommendations.append("Material delay detected: Prioritize non-dependent construction activities.")
    
    if weather == 1:
        recommendations.append("Rainy conditions: Adjust outdoor scheduling and protect materials.")
    
    if weather == 2:
        recommendations.append("High humidity: Extend curing buffer and monitor structural processes.")
    
    if complexity == 3:
        recommendations.append("High site complexity: Deploy advanced planning and real-time monitoring.")
    
    if not recommendations:
        recommendations.append("System Stable: Continue with current optimized construction plan.")

    return recommendations
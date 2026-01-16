def calculate_body_ratios(m):
    return {
        "waist_to_shoulder": m["waist_width"] / m["shoulder_width"],
        "hip_to_shoulder": m["hip_width"] / m["shoulder_width"],
        "torso_to_shoulder": m["torso_height"] / m["shoulder_width"],
    }
def clean_attribute(value):
    # Clean attribute values
    if value is None:
        return None

    value = str(value).strip()

    if value == "" or value.lower() == "nan":
        return None

    return value


def exact_match(value1, value2):
    # Compare two attribute values exactly
    value1 = clean_attribute(value1)
    value2 = clean_attribute(value2)

    if value1 is None or value2 is None:
        return None

    return 1 if value1 == value2 else 0
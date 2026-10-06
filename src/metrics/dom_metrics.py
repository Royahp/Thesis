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

def dom_attributes_similarity(widget1, widget2):
    # Compare Android widget attributes
    attributes = [
        "clickable",
        "enabled",
        "focusable",
        "checkable",
        "checked",
        "scrollable",
        "selected",
        "long-clickable",
        "password",
    ]

    matches = []
    
    for attribute in attributes:
        value1 = clean_attribute(widget1.get(attribute))
        value2 = clean_attribute(widget2.get(attribute))

        if value1 is not None and value2 is not None:
            matches.append(1 if value1 == value2 else 0)

    if not matches:
        return None

    return sum(matches) / len(matches)
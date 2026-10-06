import xml.etree.ElementTree as ET


def load_xml(xml_path):
    # Load the XML file
    tree = ET.parse(xml_path)
    return tree.getroot()


def clean(value):
    # Clean XML or CSV values
    if value is None:
        return ""

    return str(value).strip()

def normalize_bounds(value):
    # Normalize bounds format
    value = clean(value)
    return value.replace(";", ",") #check the bounds in csv and xml are written in a same way not like [90;0][240;132] and [90;0][240;132]

def find_xml_node(root, csv_widget):
    # Find the matching widget inside the XML

    resource_id = clean(csv_widget.get("resource-id"))
    widget_class = clean(csv_widget.get("class"))
    text = clean(csv_widget.get("text"))
    bounds = normalize_bounds(csv_widget.get("bounds"))

    best_node = None
    best_score = -1

    for node in root.iter("node"):

        score = 0

        if resource_id and clean(node.attrib.get("resource-id")) == resource_id:
            score += 4

        if widget_class and clean(node.attrib.get("class")) == widget_class:
            score += 2

        if text and clean(node.attrib.get("text")) == text:
            score += 2

        if bounds and clean(node.attrib.get("bounds")) == bounds:
            score += 4

        if score > best_score:
            best_score = score
            best_node = node

    if best_score <= 0:
        return None

    return best_node


def xml_node_to_tree_text(node):
    # Convert the XML subtree for APTED

    label = clean(node.attrib.get("class"))

    if not label:
        label = "node"

    label = label.replace("{", "").replace("}", "")

    children_text = ""

    for child in node:
        if child.tag == "node":
            children_text += xml_node_to_tree_text(child)

    return "{" + label + children_text + "}"
def get_node_xpath(root, target_node):
    # Build the XML hierarchy path of a widget

    def search(current_node, path):
        if current_node is target_node:
            return path

        children = [
            child for child in current_node
            if child.tag == "node"
        ]

        for index, child in enumerate(children):
            result = search(
                child,
                path + f"/node[{index + 1}]"
            )

            if result is not None:
                return result

        return None

    return search(root, "/hierarchy")
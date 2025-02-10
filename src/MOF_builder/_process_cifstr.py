import re 
import numpy as np

def remove_blank_space(value):
    return re.sub("\s", "", value)

def remove_empty_lines(lines):
    newlines = []
    for i in range(len(lines)):
        if lines[i].strip() != "":
            newlines.append(lines[i])
    return newlines

def add_quotes(value):
    return "'" + value + "'"

def remove_note_lines(lines):
    newlines = []
    for i in range(len(lines)):
        m = re.search("_", lines[i])
        if m is None:
            newlines.append(lines[i])
    return newlines


def extract_quote_lines(lines):
    newlines = []
    for i in range(len(lines)):
        if lines[i].strip()[0] == "'":
            newlines.append(lines[i])
    return newlines

def extract_xyz_lines(lines):
    newlines = []
    for i in range(len(lines)):
        if lines[i].strip()[0] != "_":
            quote_value = add_quotes(remove_blank_space(lines[i]).strip())
            newlines.append(quote_value)
        newlines = remove_empty_lines(newlines)
    return newlines


def remove_bracket(value):
    value_float = float(re.sub("\(.*?\)", "", value))
    return value_float


def remove_tail_number(value):
    return re.sub("\d", "", value)


def remove_quotes(value):
    pattern = r"[\"']([^\"']+)[\"']"
    extracted_values = re.findall(pattern, value)
    return extracted_values[0]


def convert_fraction_to_decimal(expression):
    # Function to replace fraction with its decimal equivalent
    # match as object of re.search
    def replace_fraction(match):
        numerator, denominator = map(int, match.groups())
        return str(numerator / denominator)

    # Regular expression to find fractions
    fraction_pattern = r"(-?\d+)/(\d+)"
    # get fraction
    converted_expression = re.sub(fraction_pattern, replace_fraction, expression)

    return converted_expression


def extract_xyz_coefficients_and_constant(expr_str):
    # Initialize coefficients and constant
    coeffs = {"x": 0, "y": 0, "z": 0}
    constant_term = 0

    # Regular expression to match terms with coefficients and variables
    pattern = r"([+-]?\d*\.?\d*)\s*([xyz])"
    matches = re.findall(pattern, expr_str)

    # Update coefficients based on matches
    for match in matches:
        coeff = match[0]
        variable = match[1]
        if coeff == "" or coeff == "+":
            coeff = 1
        elif coeff == "-":
            coeff = -1
        else:
            coeff = float(coeff)
        coeffs[variable] += coeff

    # match if no constant at tail
    if re.search(r"[a-zA-Z]$", expr_str):
        constant_term = 0
    else:
        # extract tail constant term
        constant_match = re.search(r"([a-zA-Z])(\d*.*)$", expr_str)
        if constant_match:
            # constant_term = constant_match.group(2)
            constant_term = convert_fraction_to_decimal(constant_match.group(2))
        else:
            constant_term = 0

    xyz_coeff_array = np.array([coeffs["x"], coeffs["y"], coeffs["z"]])

    return xyz_coeff_array, constant_term

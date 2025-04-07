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


def extract_value_str_slice(s):
    if len(s) == 0:
        return 0
    sign = 1
    mul_value = 1
    s_list = list(s[0])

    if "-" in s_list:
        sign = -1
    if "*" in s_list:
        mul_value = s_list[s_list.index("*") - 1]

    return sign * int(mul_value)


def extract_value_from_str(s):
    s = re.sub(r" ", "", s)  # remove space
    s = re.sub(r"(?<=[+-])", ",", s[::-1])[::-1]
    if s[0] == ",":
        s = s[1:]
    s_list = list(s.split(","))
    # find the slice of x
    x_slice = [s_list[i] for i in range(len(s_list)) if "x" in s_list[i]]
    y_slice = [s_list[i] for i in range(len(s_list)) if "y" in s_list[i]]
    z_slice = [s_list[i] for i in range(len(s_list)) if "z" in s_list[i]]
    # find the only digit in the slice no x,y,z
    const_slice = [
        s_list[i]
        for i in range(len(s_list))
        if "x" not in s_list[i] and "y" not in s_list[i] and "z" not in s_list[i]
    ]
    # extract the coefficient and constant from slice
    # if * exist then use the value before *, if - exist then *-1
    x_coeff = extract_value_str_slice(x_slice)
    y_coeff = extract_value_str_slice(y_slice)
    z_coeff = extract_value_str_slice(z_slice)
    if len(const_slice) == 0:
        const = 0
    else:
        const = const_slice[0]
        const = convert_fraction_to_decimal(const)

    return x_coeff, y_coeff, z_coeff, const


def extract_transformation_matrix_from_symmetry_operator(expr_str):
    expr_str = str(expr_str)
    expr_str = expr_str.strip("\n")
    expr_str = expr_str.replace(" ", "")
    split_str = expr_str.split(",")
    transformation_matrix = np.zeros((4, 4))
    transformation_matrix[3, 3] = 1
    for i in range(len(split_str)):
        x_coeff, y_coeff, z_coeff, const = extract_value_from_str(split_str[i])
        transformation_matrix[i] = [x_coeff, y_coeff, z_coeff, const]

    return transformation_matrix

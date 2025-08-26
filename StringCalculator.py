import re

# CORRECTED: The regex now correctly finds content inside square brackets.
# It looks for a literal '[', captures one or more characters (.+?),
# and then finds a literal ']'.
pattern = r'\[(.+?)\]'


class StringCalculator:
    def add(self, numbers):
        if not numbers:
            return 0

        # Default delimiters
        all_delimiters = [',', '\n']

        # Parse custom delimiters
        if numbers.startswith('//'):
            delimiter_end = numbers.index('\n')
            delimiter_section = numbers[2:delimiter_end]
            numbers = numbers[delimiter_end + 1:]

            # Extract delimiters from brackets or use single character
            if delimiter_section.startswith('['):
                # Using the corrected regex pattern
                custom_delimiters = re.findall(pattern, delimiter_section)
                all_delimiters.extend(custom_delimiters)
            else:
                all_delimiters.append(delimiter_section)

        # CORRECTED: Use re.split() for robust splitting with all delimiters
        # 1. We escape each delimiter to handle special regex characters (like '*')
        # 2. We join them with '|' (OR) to create one master pattern
        split_pattern = '|'.join(map(re.escape, all_delimiters))
        parts = re.split(split_pattern, numbers)

        # Filter out any empty strings that result from splitting
        parts = [part.strip() for part in parts if part.strip()]

        # Process numbers
        negatives = []
        total = 0

        for part in parts:
            number = int(part)
            if number < 0:
                negatives.append(number)
            elif number <= 1000:
                total += number

        if negatives:
            raise ValueError(f"negatives not allowed: {','.join(map(str, negatives))}")

        return total


# Test the functionality
def test_string_calculator():
    calc = StringCalculator()

    # Basic tests
    print("Testing basic functionality...")
    assert calc.add("") == 0
    print("✓ Empty string")

    assert calc.add("1") == 1
    print("✓ Single number")

    assert calc.add("1,2") == 3
    print("✓ Two numbers")

    assert calc.add("1,2,3") == 6
    print("✓ Multiple numbers")

    # Newline tests
    print("\nTesting newlines...")
    assert calc.add("1\n2,3") == 6
    print("✓ Newlines as delimiters")

    # Custom delimiter tests
    print("\nTesting custom delimiters...")
    assert calc.add("//;\n1;2") == 3
    print("✓ Custom single delimiter")

    # Negative number tests
    print("\nTesting negative numbers...")
    try:
        calc.add("1,-2,3")
        assert False, "Should have raised exception"
    except ValueError as e:
        assert "negatives not allowed: -2" in str(e)
        print("✓ Negative numbers throw exception")

    # Large number tests
    print("\nTesting large numbers (>1000)...")
    assert calc.add("2,1001") == 2
    print("✓ Numbers > 1000 ignored")

    # Long delimiter tests
    print("\nTesting long delimiters...")
    result = calc.add("//[***]\n1***2***3")
    print(f"Long delimiter result: {result}")
    assert result == 6
    print("✓ Long delimiters work")

    # Multiple delimiter tests
    print("\nTesting multiple delimiters...")
    assert calc.add("//[*][%]\n1*2%3") == 6
    print("✓ Multiple delimiters work")

    assert calc.add("//[***][%%%]\n1***2%%%3") == 6
    print("✓ Multiple long delimiters work")

    print("\n🎉 All tests passed!")


if __name__ == "__main__":
    test_string_calculator()
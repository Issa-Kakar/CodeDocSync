"""Main entry point for complex project."""

from package.submodule import DataProcessor
from package.utils import format_output, validate_input


def main(input_file: str, output_file: str) -> None:
    """Process data from input file and save to output file.

    Args:
        input_file: Path to input data file
        output_file: Path for processed output

    Returns:
        None
    """
    # Validate input
    if not validate_input(input_file):
        raise ValueError(f"Invalid input file: {input_file}")

    # Process data
    processor = DataProcessor()
    with open(input_file) as f:
        data = f.read()

    result = processor.process(data)
    formatted = format_output(result)

    # Save output
    with open(output_file, "w") as f:
        f.write(formatted)


def run_batch_processing(file_list: list[str]) -> dict[str, bool]:
    """Process multiple files in batch.

    Args:
        file_list: List of file paths to process

    Returns:
        Dictionary mapping filenames to success status
    """
    results = {}
    for file_path in file_list:
        try:
            output_path = file_path.replace(".txt", "_processed.txt")
            main(file_path, output_path)
            results[file_path] = True
        except Exception:
            results[file_path] = False
    return results

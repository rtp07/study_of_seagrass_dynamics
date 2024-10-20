import nbformat
from nbconvert import PDFExporter
from nbconvert.preprocessors import ExecutePreprocessor
import os
import asyncio
from asyncio import WindowsSelectorEventLoopPolicy
from datetime import datetime
import gc

# Set the WindowsSelectorEventLoopPolicy to avoid the warning
asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

def log_time(message):
    """Helper function to log messages with a timestamp."""
    print(f"{datetime.now()}: {message}")

def run_notebook(input_notebook, output_pdf, user_year):
    try:
        log_time(f"Running notebook for year {user_year}...")
        
        # Read notebook file
        with open(input_notebook, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Update the user_year variable in the notebook
        log_time(f"Updating notebook with user_year = {user_year}...")
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                source = cell['source']
                if 'user_year =' in source:
                    source = source.replace('user_year = "default"', f'user_year = "{user_year}"')
                    cell['source'] = source
        
        log_time(f"Executing the notebook for year {user_year}...")
        ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')
        ep.preprocess(notebook, {'metadata': {'path': './'}})
        
        log_time(f"Converting the notebook to PDF for year {user_year}...")
        pdf_exporter = PDFExporter()
        pdf_exporter.exclude_input = False
        
        # Convert notebook to PDF
        body, resources = pdf_exporter.from_notebook_node(notebook)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_pdf)
        os.makedirs(output_dir, exist_ok=True)

        # Write PDF to file
        with open(output_pdf, 'wb') as pdf_out:
            pdf_out.write(body)
        
        log_time(f"Successfully exported {output_pdf} for year {user_year}")
    
    except Exception as e:
        log_time(f"An error occurred for year {user_year}: {e}")
    
    finally:
        # Explicitly call garbage collection to free up memory
        gc.collect()

# List of years to process
years = [
    '2016', '2017', '2018', '2019', 
    '2020', '2021', '2022', '2023'
]

# Directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Path to the input notebook
input_nb = os.path.abspath(
    os.path.join(
        script_dir, '..', 'nb', '20240912_benthic_mapping.ipynb'
    )
)

# Process each year
for year in years:
    output_pdf = os.path.abspath(
        os.path.join(
            script_dir, '..', '..', 'docs', 'reports', 'reports_20241016', f'notebook_output_{year}.pdf'
        )
    )
    log_time(f"Preparing to process year {year}...")
    log_time(f"Output PDF path: {output_pdf}")
    run_notebook(input_nb, output_pdf, year)
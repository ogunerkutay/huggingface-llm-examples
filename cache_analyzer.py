# This script analyzes the Hugging Face Hub cache directory and provides a summary of the cached models and their sizes.
# It uses the `huggingface_hub` library to scan the cache directory and the `rich` library to display the information in a user-friendly format.

from huggingface_hub import scan_cache_dir # Import the scan_cache_dir function from the huggingface_hub library. This function is used to scan the Hugging Face cache directory.
from tabulate import tabulate # Import the tabulate function from the tabulate library. This function is used to format data into tables.
from rich.console import Console # Import the Console class from the rich.console module. This class is used to create a console object for styled output.
from rich.table import Table # Import the Table class from the rich.table module. This class is used to create tables for displaying data.
from rich.panel import Panel # Import the Panel class from the rich.panel module. This class is used to create panels for grouping related information.
from rich import box # Import the box module from the rich library. This module provides different box styles for tables and panels.

def format_size(size_bytes):
    """Convert bytes to human-readable format"""
    # This function takes a size in bytes as input and returns a human-readable string representation of the size (e.g., "1.23 MB").
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']: # Iterate through the units of size (bytes, kilobytes, megabytes, gigabytes, terabytes).
        if size_bytes < 1024.0: # If the size is less than 1024 bytes, format it with the current unit and return.
            return f"{size_bytes:.2f} {unit}" # Format the size with two decimal places and the unit.
        size_bytes /= 1024.0 # Divide the size by 1024 to convert it to the next larger unit.

def main():
    # This is the main function of the script. It scans the Hugging Face cache directory, gathers information about the cached models, and displays the information in a styled format using the `rich` library.
    try:
        # Get cache information
        print("Scanning Hugging Face cache directory...") # Print a message to the console indicating that the cache directory is being scanned.
        cache_info = scan_cache_dir() # Scan the Hugging Face cache directory and store the information in the `cache_info` variable.
        
        # Create a rich console for styled output
        console = Console() # Create a Console object for styled output.
        
        # Display summary information
        console.print(Panel.fit( # Print a panel with the title "Hugging Face Cache Summary".
            "[bold cyan]Hugging Face Cache Summary[/bold cyan]",  # The title of the panel, styled with bold and cyan color.
            border_style="cyan",  # The border style of the panel, set to cyan.
            padding=(1, 2) # The padding around the title, set to 1 line above and below, and 2 spaces on each side.
        ))
        
        summary_table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED) # Create a table for displaying the summary information.
        summary_table.add_column("Total Size") # Add a column for the total size of the cached models.
        summary_table.add_column("# Models") # Add a column for the number of cached models.
        summary_table.add_column("# Revisions") # Add a column for the number of revisions of the cached models.
        summary_table.add_column("# Files") # Add a column for the number of files in the cached models.
        
        summary_table.add_row( # Add a row to the table with the summary information.
            format_size(cache_info.size_on_disk), # The total size of the cached models, formatted using the `format_size` function.
            str(len(cache_info.repos)), # The number of cached models.
            str(sum(len(repo.revisions) for repo in cache_info.repos)), # The number of revisions of the cached models.
            str(sum(sum(len(rev.files) for rev in repo.revisions) for repo in cache_info.repos)) # The number of files in the cached models.
        )
        
        console.print(summary_table) # Print the summary table to the console.
        console.print() # Print a blank line to the console.
        
        # Display detailed information for each model
        console.print(Panel.fit( # Print a panel with the title "Models in Cache".
            "[bold green]Models in Cache[/bold green]",  # The title of the panel, styled with bold and green color.
            border_style="green",  # The border style of the panel, set to green.
            padding=(1, 2) # The padding around the title, set to 1 line above and below, and 2 spaces on each side.
        ))
        
        for repo in cache_info.repos: # Iterate through the cached models.
            # Create a table for each repository
            repo_table = Table(show_header=True, header_style="bold blue", box=box.SIMPLE) # Create a table for displaying the information about the current model.
            repo_table.add_column("Revision") # Add a column for the revision of the model.
            repo_table.add_column("Size") # Add a column for the size of the model.
            repo_table.add_column("Last Modified") # Add a column for the last modified date of the model.
            repo_table.add_column("Files") # Add a column for the number of files in the model.
            
            for revision in repo.revisions: # Iterate through the revisions of the current model.
                # Format the datetime or timestamp
                last_modified = "Unknown" # Initialize the `last_modified` variable to "Unknown".
                if revision.last_modified: # If the revision has a last modified date.
                    try:
                        # Try to handle it as a datetime object
                        if hasattr(revision.last_modified, 'strftime'): # If the last modified date is a datetime object.
                            last_modified = revision.last_modified.strftime("%Y-%m-%d %H:%M:%S") # Format the last modified date as a string.
                        # Try to handle it as a timestamp
                        else: # If the last modified date is a timestamp.
                            from datetime import datetime # Import the datetime class from the datetime module.
                            last_modified = datetime.fromtimestamp(revision.last_modified).strftime("%Y-%m-%d %H:%M:%S") # Convert the timestamp to a datetime object and format it as a string.
                    except:
                        last_modified = str(revision.last_modified) # If the last modified date cannot be formatted, convert it to a string.
                
                repo_table.add_row( # Add a row to the table with the information about the current revision.
                    revision.commit_hash[:8],  # Show first 8 chars of commit hash # The commit hash of the revision, truncated to the first 8 characters.
                    format_size(revision.size_on_disk), # The size of the revision, formatted using the `format_size` function.
                    last_modified, # The last modified date of the revision.
                    str(len(revision.files)) # The number of files in the revision.
                )
            
            # Display the repository information
            console.print(Panel.fit( # Print a panel with the name of the model.
                f"[bold yellow]{repo.repo_id}[/bold yellow]",  # The name of the model, styled with bold and yellow color.
                border_style="yellow",  # The border style of the panel, set to yellow.
                padding=(1, 1) # The padding around the name, set to 1 line above and below, and 1 space on each side.
            ))
            console.print(repo_table) # Print the table with the information about the model.
            console.print() # Print a blank line to the console.
        
    except Exception as e: # If an error occurs.
        print(f"Error: {e}") # Print the error message to the console.
        print("If you don't have rich or tabulate installed, install them with:") # Print a message to the console.
        print("pip install rich tabulate") # Print a message to the console.

if __name__ == "__main__":
    main() # Call the main function if the script is run directly.
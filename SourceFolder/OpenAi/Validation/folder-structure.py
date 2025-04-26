
import os
from pathlib import Path

def print_directory_structure(startpath: str, output_file: str = 'folder_structure.txt'):
    """
    Generate a visual tree structure of the directory and save to a file.
    
    Args:
        startpath: Root directory to start from
        output_file: Output file name to save the structure
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for root, dirs, files in os.walk(startpath):
            level = root.replace(startpath, '').count(os.sep)
            indent = '│   ' * (level - 1) + '├── ' if level > 0 else ''
            f.write(f'{indent}{os.path.basename(root)}/\n')
            
            subindent = '│   ' * level + '├── '
            
            # Sort directories and files
            dirs.sort()
            files.sort()
            
            # Print files
            for i, file in enumerate(files):
                if i == len(files) - 1 and len(dirs) == 0:
                    subindent = '│   ' * level + '└── '
                f.write(f'{subindent}{file}\n')

def get_directory_info(startpath: str) -> dict:
    """
    Get information about the directory contents.
    
    Args:
        startpath: Root directory to analyze
    
    Returns:
        dict: Statistics about the directory
    """
    info = {
        'total_files': 0,
        'total_dirs': 0,
        'file_types': {},
        'largest_files': [],
        'empty_dirs': []
    }
    
    for root, dirs, files in os.walk(startpath):
        # Count directories
        info['total_dirs'] += len(dirs)
        
        # Check for empty directories
        if not dirs and not files:
            info['empty_dirs'].append(root)
        
        # Process files
        for file in files:
            info['total_files'] += 1
            
            # Count file types
            ext = os.path.splitext(file)[1].lower()
            info['file_types'][ext] = info['file_types'].get(ext, 0) + 1
            
            # Track large files
            file_path = os.path.join(root, file)
            try:
                size = os.path.getsize(file_path)
                info['largest_files'].append((file_path, size))
            except:
                continue
    
    # Keep only top 10 largest files
    info['largest_files'].sort(key=lambda x: x[1], reverse=True)
    info['largest_files'] = info['largest_files'][:10]
    
    return info

def main():
    # Get the current directory
    current_dir = os.getcwd()
    
    print(f"Analyzing directory: {current_dir}")
    
    # Generate structure file
    print("\nGenerating directory structure...")
    print_directory_structure(current_dir)
    
    # Get directory information
    print("\nGathering directory information...")
    info = get_directory_info(current_dir)
    
    # Write detailed report
    with open('../Topic Splitter/directory_report.txt', 'w', encoding='utf-8') as f:
        f.write("=== Directory Analysis Report ===\n\n")
        
        f.write("General Statistics:\n")
        f.write(f"Total Files: {info['total_files']}\n")
        f.write(f"Total Directories: {info['total_dirs']}\n\n")
        
        f.write("File Types:\n")
        for ext, count in sorted(info['file_types'].items()):
            if ext:
                f.write(f"{ext}: {count} files\n")
            else:
                f.write(f"No extension: {count} files\n")
        f.write("\n")
        
        f.write("Largest Files:\n")
        for path, size in info['largest_files']:
            size_mb = size / (1024 * 1024)
            f.write(f"{size_mb:.2f} MB - {path}\n")
        f.write("\n")
        
        if info['empty_dirs']:
            f.write("Empty Directories:\n")
            for dir_path in info['empty_dirs']:
                f.write(f"{dir_path}\n")
    
    print("\nDone! Check 'folder_structure.txt' and 'directory_report.txt' for details.")

if __name__ == "__main__":
    main()

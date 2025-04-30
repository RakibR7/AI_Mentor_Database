# Installation and Usage for Wikipedia Dumps

This guide provides step-by-step instructions for installing and using WikiExtractor to process Wikipedia dumps.

## Prerequisites
1. Python 3.9 or higher
2. WSL (Windows Subsystem for Linux) for seamless extraction

---

## Instructions for Extracting Wikipedia Dumps

### Example Command

Run the following command to extract Wikipedia dumps using WikiExtractor:

```bash
python -m wikiextractor.WikiExtractor "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\enwiki-20240901-pages-articles-multistream.xml.bz2" \
  -o "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\Wiki_extracted_data" --json
```

---

## Setting Up WSL for Wikipedia Dump Extraction

### Step 1: Install Python 3.9 on WSL

1. **Verify Python Version**:
   ```bash
   python3 --version
   ```

2. **Update Package List**:
   ```bash
   sudo apt update
   ```

3. **Install Required Tools**:
   ```bash
   sudo apt install software-properties-common
   ```

4. **Add Python PPA**:
   ```bash
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   ```

5. **Install Python 3.9**:
   ```bash
   sudo apt install python3.9 python3.9-venv python3.9-dev
   python3.9 --version
   ```

6. **Install Pip**:
   ```bash
   sudo apt install python3-pip
   pip3 --version
   ```

### Step 2: Install pipx and WikiExtractor

1. **Install pipx**:
   ```bash
   sudo apt install pipx
   export PATH=$PATH:~/.local/bin
   echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
   source ~/.bashrc
   ```

2. **Install WikiExtractor**:
   ```bash
   pipx install wikiextractor --python python3.9
   ```

---

## Running WikiExtractor with WSL

Run the following command in your WSL terminal:

```bash
wikiextractor \
  "/mnt/c/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/enwiki-20240901-pages-articles-multistream.xml.bz2" \
  -o "/mnt/c/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/Wiki_extracted_data" --json
```

---

### Notes
- Ensure that file paths in WSL use the `/mnt/c/` prefix for Windows drives.
- The `--json` flag outputs the extracted data in JSON format for easy parsing.



# WikiExtractor Setup Guide on WSL

Follow these minimal steps to install and run **WikiExtractor** in **WSL (Windows Subsystem for Linux)** to extract Wikipedia data effectively.

### Prerequisites
- Ensure WSL is installed and configured with an Ubuntu distribution.

### Step 1: Set Up Dependencies

1. **Update the Package List and Install Necessary Software**
    ```bash
    sudo apt update
    sudo apt install software-properties-common
    ```
2. **Add the Deadsnakes PPA to Get Python 3.9**
    ```bash
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt update
    ```

### Step 2: Install Python and Required Tools

3. **Install Python 3.9, venv, and Development Headers**
    ```bash
    sudo apt install python3.9 python3.9-venv python3.9-dev
    ```
4. **Install `pip` for Python 3**
    ```bash
    sudo apt install python3-pip
    ```
5. **Install `pipx` via `apt`**
    ```bash
    sudo apt install pipx
    ```

### Step 3: Configure `pipx` and Install WikiExtractor

6. **Add `pipx` to the System Path and Make the Change Permanent**
    ```bash
    export PATH=$PATH:~/.local/bin
    echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
    source ~/.bashrc
    ```
7. **Use `pipx` to Install WikiExtractor with Python 3.9**
    ```bash
    pipx install wikiextractor --python python3.9
    ```

### Step 4: Extract Wikipedia Data

8. **Run WikiExtractor on Wikipedia Dump File**
    ```bash
    wikiextractor "/mnt/c/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/enwiki-20240901-pages-articles-multistream.xml.bz2" -o "/mnt/c/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/Wiki_extracted_data" --json
    ```

### Notes
- Ensure you replace the paths with your own file locations as necessary.
- This guide is tailored for Ubuntu in WSL and may need adjustment for different Linux distributions.

With these commands, you should be able to set up and run WikiExtractor successfully. 







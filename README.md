install for wikipedia dumps

python -m wikiextractor.WikiExtractor "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\enwiki-20240901-pages-articles-multistream.xml.bz2" -o "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\Wiki_extracted_data" --json

example below:

python -m wikiextractor.WikiExtractor "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\enwiki-20240901-pages-articles-multistream.xml.bz2" -o "C:\Users\Rakib\Documents\Ai Mentor\SourceFolder\OpenAi\Wikipedia_dumps\Wiki_extracted_data" --json

Install WSL (to extract wiki dumps)

python3 --version
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.9 python3.9-venv python3.9-dev
python3.9 --version
sudo apt install python3-pip
pip3 --version
sudo apt update
sudo apt install pipx
export PATH=$PATH:~/.local/bin
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
source ~/.bashrc
pipx install wikiextractor --python python3.9

then finally...
wikiextractor "/mnt/c/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/enwiki-20240901-pages-articles-multistream.xml.bz2" -o "/mnt/c/Users/Rakib/Documents/Ai Mentor/SourceFolder/OpenAi/Wikipedia_dumps/Wiki_extracted_data" --json




Alternative version:


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

With these commands, you should be able to set up and run WikiExtractor successfully. If you run into any issues, feel free to reach out for help! 😊







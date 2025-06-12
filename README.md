# Git and GitHub Setup Guide

## Key Terms

**Git** is a command-line tool that allows you to perform *source control*, which is a system for tracking and managing changes to code and files over time.

**GitHub** is a web-based platform that hosts Git repositories in the cloud. It provides a remote location to store your Git repositories, enables collaboration through features like pull requests and issues, and offers additional tools for project management, code review, and deployment.

**Repository (Repo)** is a directory or folder that contains your project files along with the complete history of changes tracked by Git.

**Commit** is a snapshot of your project at a specific point in time, containing the changes you've made along with a message describing what was changed.

**Branch** is a parallel version of your repository that allows you to work on different features or experiments without affecting the main codebase.

**Remote** is a version of your repository that is hosted on a server (like GitHub) rather than on your local machine.

**SSH Key** is a cryptographic key pair that provides a secure way to authenticate with GitHub without entering your username and password each time.

## Installation Notes

**Important**: Commands formatted like `git commit -m "this is the message"` are complete commands that should be pasted into the terminal exactly as shown. Be careful not to add any additional whitespace or other characters when pasting commands.

**Tip**: If "Cmd + V" (paste) doesn't work in the terminal, try "Cmd + Shift + V". The same applies for "Cmd + C" (copy).

## Installation Guide

### Step 0: Open Terminal
Open the 'Terminal' application on your Mac. All commands will be pasted 'as-is', and then you will simply press 'Enter'.

### Step 1: Install Git

First, install Homebrew (a package manager for Mac):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then install Git:
```bash
brew install git
```

**Verify Installation:**
```bash
git --version
```

### Step 2: Configure Git

Set up your global Git configuration with your name and email:

```bash
# Set your name
git config --global user.name "Your Full Name"

# Set your email (use the same email as your GitHub account)
git config --global user.email "your.email@example.com"

# Optional: Set default branch name to 'main'
git config --global init.defaultBranch main

# View your configuration
git config --list
```

### Step 3: Generate SSH Key

**Generate a New SSH Key:**
```bash
# Generate SSH key (replace with your GitHub email)
ssh-keygen -t ed25519 -C "your.email@example.com"
```

When prompted:
- **File location**: Press Enter to use default location (`~/.ssh/id_ed25519`)
- **Passphrase**: Press Enter for no passphrase

**Start SSH Agent and Add Key:**
```bash
# Start the SSH agent
eval "$(ssh-agent -s)"

# Add SSH key to ssh-agent
ssh-add ~/.ssh/id_ed25519
```

### Step 4: Add SSH Key to GitHub

**Copy Your Public Key:**
```bash
# Print your public key, then copy it (it will be a very long string of characters)
cat ~/.ssh/id_ed25519.pub
```

**Add Key to GitHub:**
1. Go to GitHub.com and sign in
2. Click your profile picture → **Settings**
3. In the left sidebar, click **SSH and GPG keys**
4. Click **New SSH key**
5. Add a descriptive title (e.g., "MacBook Pro")
6. Paste your key into the "Key" field
7. Click **Add SSH key**

**Test SSH Connection in Terminal:**
```bash
ssh -T git@github.com
```

You should see a message like: "Hi [*username*]! You've successfully authenticated..."

### Step 5: Working with Remote Repositories

**Clone an Existing Repository:**
```bash
# Clone using SSH 
git clone git@github.com:grigdon/agentic_onset.git

# Navigate to the repository
cd agentic_onset
```
Congratulations! You just cloned the repository from the command line interface! The repository is now located at `~/agentic_onset/`, assuming everything went well.

### Some helpful links

- https://www.unixtutorial.org/basic-unix-commands/
- https://education.github.com/git-cheat-sheet-education.pdf

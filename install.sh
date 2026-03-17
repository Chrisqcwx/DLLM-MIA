#!/bin/bash

################################################################################
# Code-CLI Setup Script for macOS & Linux
# Installs and configures: Node.js, npm, Git, Claude Code, and Codex
#
# Version: 1.0.0
# Author: Code-CLI Team
# License: MIT
#
# Usage: sudo bash install.sh
################################################################################

set -euo pipefail
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Color definitions for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

# Global variables
# Fix BASH_SOURCE for piped execution
if [[ -n "${BASH_SOURCE[0]}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR="${PWD}"
fi
OS_TYPE=""
PACKAGE_MANAGER=""
SHELL_CONFIG_FILE=""
SHELL_PROFILE_FILE=""

################################################################################
# 1.3 Permission Check
################################################################################

check_sudo() {
  if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}[!] This script must be run with sudo (administrator) privileges${NC}"
    echo -e "${YELLOW}[*] Please run: sudo bash install.sh${NC}"
    exit 1
  fi
}

################################################################################
# 1.4 OS Detection
################################################################################

detect_os() {
  case "$(uname -s)" in
    Darwin)
      OS_TYPE="macos"
      PACKAGE_MANAGER="homebrew"
      # For macOS, prefer .zshrc (default shell since Catalina)
      if [[ -f "$HOME/.zshrc" ]]; then
        SHELL_CONFIG_FILE="$HOME/.zshrc"
      else
        SHELL_CONFIG_FILE="$HOME/.bash_profile"
      fi
      SHELL_PROFILE_FILE="$HOME/.zshrc"
      ;;
    Linux)
      OS_TYPE="linux"
      # Detect Linux distribution's package manager
      if command -v apt-get &> /dev/null; then
        PACKAGE_MANAGER="apt"
      elif command -v yum &> /dev/null; then
        PACKAGE_MANAGER="yum"
      elif command -v dnf &> /dev/null; then
        PACKAGE_MANAGER="dnf"
      else
        echo -e "${RED}[!] Unsupported Linux distribution${NC}"
        echo -e "${YELLOW}[*] Supported: Ubuntu/Debian (apt), CentOS/RHEL (yum), Fedora (dnf)${NC}"
        exit 1
      fi
      SHELL_CONFIG_FILE="$HOME/.bashrc"
      SHELL_PROFILE_FILE="$HOME/.bashrc"
      ;;
    *)
      echo -e "${RED}[!] Unsupported operating system: $(uname -s)${NC}"
      exit 1
      ;;
  esac
}

################################################################################
# 2.1 Color & Display Functions
################################################################################

print_banner() {
  clear

  # Fixed border using ASCII characters (compatible with all terminals)
  printf "+------------------------------------------------------------------+\n"
  printf "|                                                                  |\n"
  printf "|                        Code-CLI                                  |\n"
  printf "|                                                                  |\n"
  printf "|              技术无壁垒,AI皆可调。                               |\n"
  printf "|                                                                  |\n"
  printf "|                        国际站                                    |\n"
  printf "|              https://www.code-cli.com/query                      |\n"
  printf "|                                                                  |\n"
  printf "|                        国内站                                    |\n"
  printf "|              https://www.code-cli.cn/query                       |\n"
  printf "|                                                                  |\n"
  printf "+------------------------------------------------------------------+\n"

  echo ""
}

print_section() {
  local title="$1"
  local spaces=$((60 - ${#title}))
  local padding=""
  for ((i=0; i<spaces; i++)); do
    padding+=" "
  done

  echo -e "${CYAN}+------------------------------------------------------------------+${NC}"
  echo -e "${CYAN}|  ${title}${padding}|${NC}"
  echo -e "${CYAN}+------------------------------------------------------------------+${NC}"
  echo ""
}

print_status() {
  local type="$1"
  local message="$2"

  case "$type" in
    ok)
      echo -e "${GREEN}[OK]${NC} $message"
      ;;
    info)
      echo -e "${CYAN}[*]${NC} $message"
      ;;
    warn)
      echo -e "${YELLOW}[!]${NC} $message"
      ;;
    error)
      echo -e "${RED}[x]${NC} $message"
      ;;
    debug)
      echo -e "${GRAY}[DEBUG]${NC} $message"
      ;;
    *)
      echo "$message"
      ;;
  esac
}

################################################################################
# 2.2 File and Path Operations
################################################################################

ensure_directory() {
  local dir="$1"
  if [[ ! -d "$dir" ]]; then
    mkdir -p "$dir"
    print_status "info" "Created directory: $dir"
  else
    print_status "info" "Found directory: $dir"
  fi
}

write_file_no_bom() {
  local file_path="$1"
  local content="$2"

  # Write file using printf to avoid BOM
  printf "%s" "$content" > "$file_path"
}

backup_file() {
  local file="$1"
  if [[ -f "$file" ]]; then
    local backup="${file}.backup"
    cp "$file" "$backup"
    print_status "info" "Created backup: $backup"
    echo "$backup"
  fi
}

################################################################################
# 2.3 Environment Variable and PATH Management
################################################################################

add_to_path() {
  local path_dir="$1"

  if [[ ":$PATH:" != *":$path_dir:"* ]]; then
    export PATH="$path_dir:$PATH"

    # Also add to shell config file for persistence
    if ! grep -q "export PATH=\"$path_dir:\$PATH\"" "$SHELL_CONFIG_FILE"; then
      echo "export PATH=\"$path_dir:\$PATH\"" >> "$SHELL_CONFIG_FILE"
      print_status "info" "Added $path_dir to PATH in $SHELL_CONFIG_FILE"
    fi
  fi
}

set_env_var() {
  local var_name="$1"
  local var_value="$2"

  # Set in current session
  export "$var_name=$var_value"

  # Persist in shell config file
  # Remove any existing definition first
  sed -i.bak "/export ${var_name}=/d" "$SHELL_CONFIG_FILE" 2>/dev/null || sed -i "" "/export ${var_name}=/d" "$SHELL_CONFIG_FILE"

  # Add new definition
  echo "export ${var_name}=\"${var_value}\"" >> "$SHELL_CONFIG_FILE"

  print_status "info" "Set $var_name in $SHELL_CONFIG_FILE"
}

################################################################################
# 2.4 Package Manager Operations
################################################################################

ensure_homebrew() {
  if command -v brew &> /dev/null; then
    print_status "ok" "Homebrew already installed"
    return 0
  fi

  print_status "info" "Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

  if command -v brew &> /dev/null; then
    print_status "ok" "Homebrew installed successfully"
    return 0
  else
    print_status "error" "Homebrew installation failed"
    return 1
  fi
}

ensure_apt() {
  if command -v apt-get &> /dev/null; then
    print_status "ok" "apt-get is available"
    print_status "info" "Running apt-get update..."
    sudo apt-get update -qq
    return 0
  else
    print_status "error" "apt-get not found"
    return 1
  fi
}

ensure_yum() {
  if command -v yum &> /dev/null; then
    print_status "ok" "yum is available"
    return 0
  elif command -v dnf &> /dev/null; then
    print_status "ok" "dnf is available"
    return 0
  else
    print_status "error" "yum/dnf not found"
    return 1
  fi
}

################################################################################
# 2.5 Application Installation Verification
################################################################################

check_command_exists() {
  local cmd="$1"
  if command -v "$cmd" &> /dev/null; then
    return 0
  else
    return 1
  fi
}

get_node_version() {
  if check_command_exists "node"; then
    node --version 2>/dev/null || echo ""
  else
    echo ""
  fi
}

get_npm_version() {
  if check_command_exists "npm"; then
    npm --version 2>/dev/null || echo ""
  else
    echo ""
  fi
}

get_git_version() {
  if check_command_exists "git"; then
    git --version 2>/dev/null | awk '{print $3}' || echo ""
  else
    echo ""
  fi
}

################################################################################
# 3.1 Main Menu
################################################################################

show_menu() {
  print_banner

  echo -e "${GREEN}请选择一个选项 (Select an option):${NC}"
  echo -e "${WHITE}  1) 安装运行时环境 (Install runtime - Node.js / npm / Git)${NC}"
  echo -e "${WHITE}  2) 安装 Claude Code (npm -g @anthropic-ai/claude-code)${NC}"
  echo -e "${WHITE}  3) 安装 Codex (npm -g @openai/codex)${NC}"
  echo -e "${WHITE}  4) 配置 Claude Code 环境 (ANTHROPIC_BASE_URL / ANTHROPIC_API_KEY)${NC}"
  echo -e "${WHITE}  5) 配置 Codex 环境 (base_url / CODE_CLI_API_KEY)${NC}"
  echo -e "${WHITE}  6) 修复连接错误 (Fix ERR_BAD_REQUEST - Claude Code connection error)${NC}"
  echo -e "${WHITE}  0) 退出 (Exit)${NC}"
  echo ""
  read -p "请输入数字 (Enter number): " choice
}

################################################################################
# 4.1-4.5 Option 1: Install Runtime Environment
################################################################################

handle_option_1() {
  print_section "Installing Runtime Environment"

  # Step 1: Check/Install Node.js
  echo ""
  print_section "Step 1: Checking Node.js"

  local node_version=$(get_node_version)
  if [[ -n "$node_version" ]]; then
    print_status "ok" "Node.js $node_version already installed"
  else
    print_status "info" "Installing Node.js..."

    if [[ "$OS_TYPE" == "macos" ]]; then
      brew install node
    elif [[ "$OS_TYPE" == "linux" ]]; then
      if [[ "$PACKAGE_MANAGER" == "apt" ]]; then
        sudo apt-get install -y nodejs npm
      elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
        sudo yum install -y nodejs npm
      elif [[ "$PACKAGE_MANAGER" == "dnf" ]]; then
        sudo dnf install -y nodejs npm
      fi
    fi

    node_version=$(get_node_version)
    if [[ -n "$node_version" ]]; then
      print_status "ok" "Node.js $node_version installed"
    else
      print_status "error" "Node.js installation failed"
      return 1
    fi
  fi

  # Step 2: Verify npm
  echo ""
  print_section "Step 2: Checking npm"

  local npm_version=$(get_npm_version)
  if [[ -n "$npm_version" ]]; then
    print_status "ok" "npm $npm_version already installed"
  else
    print_status "error" "npm not found"
    return 1
  fi

  # Step 3: Check/Install Git
  echo ""
  print_section "Step 3: Checking Git"

  local git_version=$(get_git_version)
  if [[ -n "$git_version" ]]; then
    print_status "ok" "Git $git_version already installed"
  else
    print_status "info" "Installing Git..."

    if [[ "$OS_TYPE" == "macos" ]]; then
      brew install git
    elif [[ "$OS_TYPE" == "linux" ]]; then
      if [[ "$PACKAGE_MANAGER" == "apt" ]]; then
        sudo apt-get install -y git
      elif [[ "$PACKAGE_MANAGER" == "yum" ]]; then
        sudo yum install -y git
      elif [[ "$PACKAGE_MANAGER" == "dnf" ]]; then
        sudo dnf install -y git
      fi
    fi

    git_version=$(get_git_version)
    if [[ -n "$git_version" ]]; then
      print_status "ok" "Git $git_version installed"
    else
      print_status "error" "Git installation failed"
      return 1
    fi
  fi

  echo ""
  print_status "ok" "Runtime environment setup complete!"
}

################################################################################
# 5.1-5.4 Option 2: Install Claude Code
################################################################################

handle_option_2() {
  print_section "Installing Claude Code"

  if ! check_command_exists "npm"; then
    print_status "error" "npm not found. Please run option 1 first."
    return 1
  fi

  print_status "info" "Installing @anthropic-ai/claude-code..."
  npm install -g @anthropic-ai/claude-code

  if check_command_exists "claude"; then
    local version=$(claude --version 2>/dev/null || echo "installed")
    print_status "ok" "Claude Code installed ($version)"
  else
    print_status "error" "Claude Code installation verification failed"
    return 1
  fi
}

################################################################################
# 6.1-6.4 Option 3: Install Codex
################################################################################

handle_option_3() {
  print_section "Installing Codex"

  if ! check_command_exists "npm"; then
    print_status "error" "npm not found. Please run option 1 first."
    return 1
  fi

  print_status "info" "Installing @openai/codex..."
  npm install -g @openai/codex

  if check_command_exists "codex"; then
    local version=$(codex --version 2>/dev/null || echo "installed")
    print_status "ok" "Codex installed ($version)"
  else
    print_status "error" "Codex installation verification failed"
    return 1
  fi
}

################################################################################
# 7.1-7.5 Option 4: Configure Claude Code Environment
################################################################################

handle_option_4() {
  print_section "Configuring Claude Code Environment"

  # Input: ANTHROPIC_API_KEY
  echo "Enter ANTHROPIC_API_KEY (input will be visible):"
  read -p "> " api_key
  if [[ -z "$api_key" ]]; then
    print_status "error" "ANTHROPIC_API_KEY is required. Aborted."
    return 1
  fi

  # Input: ANTHROPIC_BASE_URL
  read -p "Enter ANTHROPIC_BASE_URL (default: https://api.anthropic.com): " base_url
  base_url="${base_url:-https://api.anthropic.com}"

  # Create/backup .claude.json
  local claude_config="$HOME/.claude.json"
  if [[ -f "$claude_config" ]]; then
    backup_file "$claude_config"
  fi

  # Build and write JSON config
  local json_content=$(cat <<EOF
{
  "ANTHROPIC_API_KEY": "${api_key}",
  "ANTHROPIC_BASE_URL": "${base_url}",
  "hasCompletedOnboarding": true
}
EOF
)

  write_file_no_bom "$claude_config" "$json_content"
  print_status "ok" "Wrote: $claude_config"

  # Set environment variables
  set_env_var "ANTHROPIC_API_KEY" "$api_key"
  set_env_var "ANTHROPIC_BASE_URL" "$base_url"

  # Verify
  print_status "info" "Configuration saved to $claude_config"
  print_status "info" "Environment variables set in $SHELL_CONFIG_FILE"
  print_status "info" "Please run: source $SHELL_CONFIG_FILE"
}

################################################################################
# 8.1-8.7 Option 5: Configure Codex Environment
################################################################################

handle_option_5() {
  print_section "Configuring Codex Environment"

  # Ensure ~/.codex exists
  local codex_dir="$HOME/.codex"
  ensure_directory "$codex_dir"

  # Input: CODE_CLI_API_KEY
  echo "Enter CODE_CLI_API_KEY (input will be visible):"
  read -p "> " api_key
  if [[ -z "$api_key" ]]; then
    print_status "error" "CODE_CLI_API_KEY is required. Aborted."
    return 1
  fi

  # Input: base_url
  read -p "Enter base_url (default: https://www.code-cli.cn/api/codex): " base_url
  base_url="${base_url:-https://www.code-cli.cn/api/codex}"

  # Create auth.json
  local auth_path="$codex_dir/auth.json"
  local auth_json=$(cat <<'EOF'
{
  "OPENAI_API_KEY": null
}
EOF
)

  write_file_no_bom "$auth_path" "$auth_json"
  print_status "ok" "Wrote: $auth_path"

  # Create config.toml
  local config_path="$codex_dir/config.toml"
  local config_toml=$(cat <<EOF
windows_wsl_setup_acknowledged = true
model_provider = "code-cli"
model = "gpt-5-codex"
model_reasoning_effort = "high"
disable_response_storage = true
preferred_auth_method = "apikey"
[model_providers.code-cli]
name = "code-cli"
base_url = "${base_url}"
wire_api = "responses"
requires_openai_auth = true
env_key = "CODE_CLI_API_KEY"
EOF
)

  write_file_no_bom "$config_path" "$config_toml"
  print_status "ok" "Wrote: $config_path"

  # Set environment variable
  set_env_var "CODE_CLI_API_KEY" "$api_key"

  # Verify
  print_status "info" "Configuration files created:"
  print_status "info" "  - $auth_path"
  print_status "info" "  - $config_path"
  print_status "info" "Environment variables set in $SHELL_CONFIG_FILE"
  print_status "info" "Please run: source $SHELL_CONFIG_FILE"
}

################################################################################
# 9.1-9.4 Option 6: Fix Connection Error
################################################################################

handle_option_6() {
  print_section "Fixing Claude Code Connection Error"

  local claude_config="$HOME/.claude.json"

  if [[ ! -f "$claude_config" ]]; then
    print_status "warn" "File not found: $claude_config"
    print_status "info" "Creating new configuration..."

    local json_content=$(cat <<'EOF'
{
  "hasCompletedOnboarding": true
}
EOF
)

    write_file_no_bom "$claude_config" "$json_content"
    print_status "ok" "Configuration file created"
    return 0
  fi

  # Backup and read existing config
  backup_file "$claude_config"

  # Update JSON to add/ensure hasCompletedOnboarding flag
  # Using a simple approach that works across platforms
  local temp_config="${claude_config}.tmp"

  # Read the existing file, modify it, and write back
  if command -v jq &> /dev/null; then
    # Use jq if available for robust JSON manipulation
    jq '.hasCompletedOnboarding = true' "$claude_config" > "$temp_config"
    mv "$temp_config" "$claude_config"
  else
    # Fallback: Read file and reconstruct with the flag
    local temp_json=$(cat "$claude_config")
    # Remove the closing brace and add the flag
    temp_json="${temp_json%\}}"
    temp_json="${temp_json%,}"
    temp_json="${temp_json},"$'\n'"  \"hasCompletedOnboarding\": true"$'\n'"}"
    write_file_no_bom "$claude_config" "$temp_json"
  fi

  print_status "ok" "Added hasCompletedOnboarding flag to configuration"
  print_status "info" "Configuration file: $claude_config"
  print_status "info" "Backup file: ${claude_config}.backup"
}

################################################################################
# Main Loop
################################################################################

main() {
  check_sudo
  detect_os

  # Initialize package manager based on OS
  case "$OS_TYPE" in
    macos)
      ensure_homebrew
      ;;
    linux)
      case "$PACKAGE_MANAGER" in
        apt) ensure_apt ;;
        yum|dnf) ensure_yum ;;
      esac
      ;;
  esac

  while true; do
    show_menu

    case "$choice" in
      1)
        handle_option_1
        ;;
      2)
        handle_option_2
        ;;
      3)
        handle_option_3
        ;;
      4)
        handle_option_4
        ;;
      5)
        handle_option_5
        ;;
      6)
        handle_option_6
        ;;
      0)
        print_status "ok" "再见! (Bye.)"
        exit 0
        ;;
      *)
        print_status "warn" "Invalid choice. Please try again."
        ;;
    esac

    echo ""
    read -p "Press Enter to continue..."
  done
}

# Run main function
main

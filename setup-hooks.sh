#!/bin/bash
# setup-hooks.sh for snomed_methods

# Set the core.hooksPath to .githooks
git config core.hooksPath .githooks

# Make the pre-push hook executable
chmod +x .githooks/pre-push

chmod +x .githooks/pre-commit

echo "Git hooks configured successfully."

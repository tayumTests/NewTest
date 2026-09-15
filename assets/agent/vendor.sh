#!/bin/bash
# Build sap-cloud-sdk wheel from git and place it in vendor/ directory.
# Uses the root-level requirements.txt to get the git URL.
#
# Usage:
#   ./vendor.sh
#
# After running, the vendor/ directory will contain the sap-cloud-sdk wheel.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR/../../../.."
VENDOR_DIR="$SCRIPT_DIR/vendor"
ROOT_REQUIREMENTS="$ROOT_DIR/requirements.txt"

echo "Building sap-cloud-sdk wheel from git..."
echo "  Using requirements from: $ROOT_REQUIREMENTS"

# Extract the sap-cloud-sdk git URL from root requirements.txt
SAP_SDK_LINE=$(grep "^sap-cloud-sdk" "$ROOT_REQUIREMENTS" || true)

if [ -z "$SAP_SDK_LINE" ]; then
    echo "Error: Could not find sap-cloud-sdk in $ROOT_REQUIREMENTS"
    exit 1
fi

echo "  Found: $SAP_SDK_LINE"

# Clean existing vendor directory
rm -rf "$VENDOR_DIR"
mkdir -p "$VENDOR_DIR"

# Create a temporary directory for building
BUILD_DIR=$(mktemp -d)
trap "rm -rf $BUILD_DIR" EXIT

# Download and build wheel from git URL
echo "  Building wheel..."
pip3 wheel \
    "$SAP_SDK_LINE" \
    --wheel-dir "$VENDOR_DIR" \
    --no-deps \
    --no-cache-dir

# Count downloaded files
COUNT=$(ls -1 "$VENDOR_DIR"/*.whl 2>/dev/null | wc -l)
echo "✓ Built $COUNT wheel(s) in vendor/"

if [ "$COUNT" -eq 0 ]; then
    echo "⚠ Warning: No wheels built. Check the git URL in requirements.txt."
    exit 1
fi

# List the wheels
echo ""
echo "Vendored wheels:"
ls -la "$VENDOR_DIR"/*.whl

echo ""
echo "✓ Done. Update agent requirements.txt to use:"
echo "  sap-cloud-sdk @ file:vendor/<wheel-filename>"

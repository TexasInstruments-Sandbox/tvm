#!/bin/bash
#
# Find IP address by MAC address on a given subnet
#
# Usage: ./find_ip_by_mac.sh <MAC_ADDRESS> [SUBNET]
# Example: ./find_ip_by_mac.sh aa:bb:cc:dd:ee:ff 192.168.1.0/24
#

MAC="${1}"
SUBNET="${2:-10.219.14.0/23}"

if [ -z "$MAC" ]; then
    echo "Usage: $0 <MAC_ADDRESS> [SUBNET]"
    echo "Example: $0 aa:bb:cc:dd:ee:ff 192.168.1.0/24"
    echo "Default subnet: 10.219.14.0/23"
    exit 1
fi

# Normalize MAC to lowercase for comparison
MAC_LOWER=$(echo "$MAC" | tr '[:upper:]' '[:lower:]')

echo "Searching for MAC $MAC_LOWER on subnet $SUBNET..."

# First check existing ARP cache
echo "Checking ARP cache..."
RESULT=$(arp -an 2>/dev/null | grep -i "$MAC_LOWER" | grep -oP '\(\K[0-9.]+(?=\))' | head -1)
if [ -n "$RESULT" ]; then
    echo "Found: $RESULT"
    exit 0
fi

# Method 1: Try arp-scan (fast and reliable)
if command -v arp-scan &>/dev/null; then
    echo "Using arp-scan..."
    # Find the interface for this subnet
    IFACE=$(ip route show "$SUBNET" 2>/dev/null | awk '{print $3}' | head -1)
    if [ -z "$IFACE" ]; then
        # Fallback: get default interface
        IFACE=$(ip route show default | awk '{print $5}' | head -1)
    fi

    if [ -n "$IFACE" ]; then
        RESULT=$(sudo arp-scan --interface="$IFACE" "$SUBNET" 2>/dev/null | grep -i "$MAC_LOWER" | awk '{print $1}' | head -1)
        if [ -n "$RESULT" ]; then
            echo "Found: $RESULT"
            exit 0
        fi
    fi
fi

# Method 2: Try nmap
if command -v nmap &>/dev/null; then
    echo "Using nmap..."
    # nmap -sn does a ping scan, populates ARP table
    sudo nmap -sn "$SUBNET" &>/dev/null

    # Now check ARP cache again
    RESULT=$(arp -an 2>/dev/null | grep -i "$MAC_LOWER" | grep -oP '\(\K[0-9.]+(?=\))' | head -1)
    if [ -n "$RESULT" ]; then
        echo "Found: $RESULT"
        exit 0
    fi

    # Also try ip neigh
    RESULT=$(ip neigh show 2>/dev/null | grep -i "$MAC_LOWER" | awk '{print $1}' | head -1)
    if [ -n "$RESULT" ]; then
        echo "Found: $RESULT"
        exit 0
    fi
fi

echo "MAC address $MAC_LOWER not found on subnet $SUBNET"
exit 1

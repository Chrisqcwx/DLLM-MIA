# Download and prepare MIMIR benchmark datasets
# python dataset/prep_mimir.py

# Prepare standard NLP datasets (wikitext, ag_news, xsum)
# python dataset/prep.py

#!/usr/bin/env bash

MOUNT=${1:-/data/sdb}
INTERVAL=${2:-5}

tmp1=$(mktemp)
tmp2=$(mktemp)

get_io() {
    for pid in $(sudo fuser -m "$MOUNT" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$'); do
        if [ -r /proc/$pid/io ]; then
            user=$(ps -o user= -p "$pid" 2>/dev/null)
            read_bytes=$(grep '^read_bytes:' /proc/$pid/io | awk '{print $2}')
            write_bytes=$(grep '^write_bytes:' /proc/$pid/io | awk '{print $2}')
            echo "$pid $user $read_bytes $write_bytes"
        fi
    done
}

get_io > "$tmp1"
sleep "$INTERVAL"
get_io > "$tmp2"

awk '
NR==FNR {
    r[$1]=$3
    w[$1]=$4
    u[$1]=$2
    next
}
{
    pid=$1
    user=$2
    dr=$3-r[pid]
    dw=$4-w[pid]
    if (dr < 0) dr=0
    if (dw < 0) dw=0
    read_sum[user]+=dr
    write_sum[user]+=dw
}
END {
    printf "%-15s %15s %15s %15s\n", "USER", "READ_MB", "WRITE_MB", "TOTAL_MB"
    for (user in read_sum) {
        read_mb=read_sum[user]/1024/1024
        write_mb=write_sum[user]/1024/1024
        total_mb=read_mb+write_mb
        printf "%-15s %15.2f %15.2f %15.2f\n", user, read_mb, write_mb, total_mb
    }
}
' "$tmp1" "$tmp2"

rm -f "$tmp1" "$tmp2"
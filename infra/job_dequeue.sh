#!/bin/bash

queue_file=$HOME/tpus/queue.txt
upper_bound_file=$HOME/tpus/upper_bound.txt
update_file=$HOME/tpus/update.txt
lock_dir=$HOME/tpus/lock
submitted_file=$HOME/tpus/submitted.txt

while true; do
    if [ -d $lock_dir ]; then
        echo "Lock directory exists"
        sleep 10
        continue
    fi

    now=`date +'%Y-%m-%d_%H-%M-%S'`
    my_128_bound=`cat $upper_bound_file | grep "128: " | cut -f2 -d' '`
    my_256_bound=`cat $upper_bound_file | grep "256: " | cut -f2 -d' '`
    my_512_bound=`cat $upper_bound_file | grep "512: " | cut -f2 -d' '`

    if [ -n "$my_128_bound" ] && [ "$my_128_bound" -eq "$my_128_bound" ] 2>/dev/null; then
        echo $now, "Current upper bound: $my_128_bound (v3-128)"
    else
        echo "Error: $my_128_bound is not a number"
        sleep 5
        continue
    fi

    if [ -n "$my_256_bound" ] && [ "$my_256_bound" -eq "$my_256_bound" ] 2>/dev/null; then
        echo $now, "Current upper bound: $my_256_bound (v3-256)"
    else
        echo "Error: $my_256_bound is not a number"
        sleep 5
        continue
    fi

    if [ -n "$my_512_bound" ] && [ "$my_512_bound" -eq "$my_512_bound" ] 2>/dev/null; then
        echo $now, "Current upper bound: $my_512_bound (v3-512)"
    else
        echo "Error: $my_512_bound is not a number"
        sleep 5
        continue
    fi

    my_current_use_128=`~/mae_jax/infra/list.sh | grep "xinleic-mae-i-" | wc -l`
    my_current_use_256=`~/mae_jax/infra/list.sh | grep "xinleic-mae-ii-" | wc -l`
    my_current_use_512=`~/mae_jax/infra/list.sh | grep "xinleic-mae-iv-" | wc -l`
    echo $now, "My current use: $my_current_use_128 (v3-128), $my_current_use_256 (v3-256), $my_current_use_512 (v3-512)"

    echo "Jobs in queue: `cat $queue_file | wc -l`"
    counter=0
    while IFS= read line; do
        if [ ! -z "$line" ]; then
            tpu_type=`echo "$line" | cut -f4 -d' '`
            if [[ $tpu_type = *"128"* ]]; then
                let "my_hypothetical_use_128 = $my_current_use_128 + 1"
                if (( $my_hypothetical_use_128 <= $my_128_bound )); then
                    # execute the line
                    echo "Submitting: $line"
                    eval "$line &"
                    echo $line >> $submitted_file
                    my_current_use_128=$my_hypothetical_use_128
                    let "counter++"
                    sleep 2
                fi
            elif [[ $tpu_type = *"256"* ]]; then
                let "my_hypothetical_use_256 = $my_current_use_256 + 1"
                if (( $my_hypothetical_use_256 <= $my_256_bound )); then
                    # execute the line
                    echo "Submitting: $line"
                    eval "$line &"
                    echo $line >> $submitted_file
                    my_current_use_256=$my_hypothetical_use_256
                    let "counter++"
                    sleep 2
                fi
            elif [[ $tpu_type = *"512"* ]]; then
                let "my_hypothetical_use_512 = $my_current_use_512 + 1"
                if (( $my_hypothetical_use_512 <= $my_512_bound )); then
                    # execute the line
                    echo "Submitting: $line"
                    eval "$line &"
                    echo $line >> $submitted_file
                    my_current_use_512=$my_hypothetical_use_512
                    let "counter++"
                    sleep 2
                fi
            fi
        fi
    done < $queue_file
    echo "Submitted $counter jobs."

    if [ ! -d $lock_dir ] && [ $counter -gt 0 ] ; then
        echo "Remove submitted jobs"
        awk 'NR==FNR{a[$0];next} !($0 in a)' $submitted_file $queue_file > $update_file
        rm $queue_file
        mv $update_file $queue_file
        echo "After cleaning: `cat $queue_file | wc -l`"
    fi

    sleep 10
done
import json

with open('latest_signals.json', 'r') as f:
    signals = json.load(f)

max_tp_long = max_sl_long = 0
max_tp_short = max_sl_short = 0
min_tp_long = min_sl_long = float('inf')
min_tp_short = min_sl_short = float('inf')

bad_tp_long = bad_sl_long = 0
bad_tp_short = bad_sl_short = 0

def pct(val):
    return f"{val:.2f}%" if val != float('inf') else "N/A"

for s in signals:
    entry = s['entry_price']
    tp = s['take_profit']
    sl = s['stop_loss']
    side = s.get('side', s.get('direction', '')).lower()
    if 'long' in side:
        tp_pct = ((tp - entry) / entry) * 100
        sl_pct = ((entry - sl) / entry) * 100
        if tp_pct <= 0:
            bad_tp_long += 1
            continue
        if sl_pct <= 0:
            bad_sl_long += 1
            continue
        if tp_pct > max_tp_long:
            max_tp_long = tp_pct
        if sl_pct > max_sl_long:
            max_sl_long = sl_pct
        if tp_pct < min_tp_long:
            min_tp_long = tp_pct
        if sl_pct < min_sl_long:
            min_sl_long = sl_pct
    elif 'short' in side:
        tp_pct = ((entry - tp) / entry) * 100
        sl_pct = ((sl - entry) / entry) * 100
        if tp_pct <= 0:
            bad_tp_short += 1
            continue
        if sl_pct <= 0:
            bad_sl_short += 1
            continue
        if tp_pct > max_tp_short:
            max_tp_short = tp_pct
        if sl_pct > max_sl_short:
            max_sl_short = sl_pct
        if tp_pct < min_tp_short:
            min_tp_short = tp_pct
        if sl_pct < min_sl_short:
            min_sl_short = sl_pct

print(f"Max TP% (long):   {pct(max_tp_long)}")
print(f"Min TP% (long):   {pct(min_tp_long)}")
print(f"Max SL% (long):   {pct(max_sl_long)}")
print(f"Min SL% (long):   {pct(min_sl_long)}")
print(f"Max TP% (short):  {pct(max_tp_short)}")
print(f"Min TP% (short):  {pct(min_tp_short)}")
print(f"Max SL% (short):  {pct(max_sl_short)}")
print(f"Min SL% (short):  {pct(min_sl_short)}")
print()
print(f"Bad TP (long): {bad_tp_long}")
print(f"Bad SL (long): {bad_sl_long}")
print(f"Bad TP (short): {bad_tp_short}")
print(f"Bad SL (short): {bad_sl_short}") 
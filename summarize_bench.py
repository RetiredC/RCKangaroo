#!/usr/bin/env python3
import sys, re, statistics, os, glob

# Extrae el máximo 'Speed: XXX MKeys/s' en cada log (robusto a warmups)
speed_re = re.compile(r"Speed:\s*([0-9]+(?:\.[0-9]+)?)\s*MKeys/s", re.IGNORECASE)

# '/usr/bin/time -f "%E real  %Mk RSS  %I in KB  %O out KB"'
time_re  = re.compile(r"(?P<real>(\d+:)?\d+\.\d+)\s+real")
rss_re   = re.compile(r"\s(?P<rss>\d+)k\s+RSS", re.IGNORECASE)

def parse_real_to_seconds(s):
    # formatos "0:28.40" ó "12.34"
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)

def parse_log(path):
    max_speed = None
    real_s = None
    rss_kb = None

    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = speed_re.search(line)
            if m:
                v = float(m.group(1))
                if (max_speed is None) or (v > max_speed):
                    max_speed = v

        f.seek(0)
        data = f.read()

        tm = time_re.search(data)
        if tm:
            real_s = parse_real_to_seconds(tm.group("real"))
        rm = rss_re.search(data)
        if rm:
            rss_kb = int(rm.group("rss"))

    return max_speed, real_s, rss_kb

def pick(val_list, func):
    vals = [v for v in val_list if v is not None]
    if not vals: return ""
    return func(vals)

def main():
    if len(sys.argv) < 2:
        print("Uso: summarize_bench.py <logdir> [pattern]", file=sys.stderr)
        sys.exit(1)
    logdir = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) >= 3 else "*.log"

    files = glob.glob(os.path.join(logdir, pattern))
    if not files:
        print("No hay logs en", logdir, file=sys.stderr)
        sys.exit(2)

    # nombres: <tag>_dp<dp>_tb<tb>_tr<tr>_runX.log
    import re
    name_re = re.compile(r".*?_dp(?P<dp>\d+)_tb(?P<tb>\d+)_tr(?P<tr>\d+)_run(?P<run>\d+)\.log$")

    groups = {}
    for p in sorted(files):
        m = name_re.match(os.path.basename(p))
        if not m:
            continue
        dp = int(m.group("dp")); tb = int(m.group("tb")); tr = int(m.group("tr"))
        metrics = parse_log(p)
        groups.setdefault((dp,tb,tr), []).append(metrics)

    print("dp,tame_bits,tame_ratio,runs,median_speed_MKeys,max_speed_MKeys,median_real_s,median_rss_kb")

    for (dp,tb,tr), lst in sorted(groups.items()):
        speeds  = [x[0] for x in lst if x[0] is not None]
        reals   = [x[1] for x in lst if x[1] is not None]
        rsslist = [x[2] for x in lst if x[2] is not None]

        med_speed = pick(speeds, statistics.median)
        max_speed = pick(speeds, max)
        med_real  = pick(reals, statistics.median)
        med_rss   = pick(rsslist, statistics.median)

        def fmt(x):
            if x == "": return ""
            if isinstance(x, float): 
                return f"{x:.3f}"
            return str(x)

        print(f"{dp},{tb},{tr},{len(lst)},{fmt(med_speed)},{fmt(max_speed)},{fmt(med_real)},{fmt(med_rss)}")

if __name__ == '__main__':
    main()

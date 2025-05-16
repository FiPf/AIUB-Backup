import datetime
import os
"""
Splits a PROOF .aut file into separate files based on specified date intervals within a year.

Parameters:
-----------
filename : str
    Full path to the input .aut file (e.g. "C:/.../GEO_2019.aut").
year : int
    The year corresponding to the data entries in the .aut file.
months : list of int
    List of split points used to define the output intervals. Valid values are:
    - 2  → Jan 1 – Mar 15
    - 5  → Mar 15 – Jun 15
    - 8  → Jun 15 – Sep 15
    - 11 → Sep 15 – Dec 31
    The user may provide a subset of these (e.g. [2, 5, 11]).
output_dir : str
    Path to the folder where the resulting split files should be saved.

Behavior:
---------
- For each specified month value in `months`, a corresponding time interval is defined.
- Each line in the input file is assigned to one of these intervals based on its date.
- If a month (e.g. 8) is missing from `months`, its time interval is *not skipped*:
    - Instead, it is divided evenly and distributed between the neighboring defined intervals.
    - This ensures no data is lost regardless of which month anchors are provided.

Output:
-------
- Creates one new .aut file for each interval corresponding to a specified or inferred month.
- Files are saved in `output_dir` and named as: original_filename_base_MM.aut
    (e.g., "GEO_2019_05.aut").

Example:
--------
split_aut_file_distribute_missing(
    filename="C:/.../GEO_2019.aut",
    year=2019,
    months=[2, 5, 11],
    output_dir="C:/.../monthly_aut_files"
)

This will create:
- GEO_2019_02.aut: Jan 1 – Mar 15
- GEO_2019_05.aut: Mar 15 – Jun 15 plus first half of Jun 15 – Sep 15
- GEO_2019_11.aut: second half of Jun 15 – Sep 15 plus Sep 15 – Dec 31
"""

import os
import datetime

def split_aut_file_distribute_missing(input_filename, year, months, output_dir):
    year = int(year)
    months = sorted(set(int(m) for m in months))
    
    # Fixed full-year intervals
    full_intervals = {
        2:  (datetime.date(year, 1, 1),  datetime.date(year, 3, 15)),
        5:  (datetime.date(year, 3, 15), datetime.date(year, 6, 15)),
        8:  (datetime.date(year, 6, 15), datetime.date(year, 9, 15)),
        11: (datetime.date(year, 9, 15), datetime.date(year, 12, 31)),
    }

    for m in months:
        if m not in full_intervals:
            raise ValueError(f"Month {m:02d} not valid. Allowed: 02, 05, 08, 11")

    assigned_intervals = {m: [] for m in months}
    all_months_sorted = sorted(full_intervals.keys())

    def neighbors(miss):
        prevs = [m for m in months if m < miss]
        nexts = [m for m in months if m > miss]
        return (max(prevs) if prevs else None), (min(nexts) if nexts else None)

    for m in all_months_sorted:
        interval = full_intervals[m]
        if m in months:
            assigned_intervals[m].append(interval)
        else:
            prev_m, next_m = neighbors(m)
            start, end = interval
            mid = start + (end - start) / 2
            if prev_m is None and next_m is None:
                continue
            elif prev_m is None:
                assigned_intervals[next_m].append(interval)
            elif next_m is None:
                assigned_intervals[prev_m].append(interval)
            else:
                assigned_intervals[prev_m].append((start, mid))
                assigned_intervals[next_m].append((mid, end))

    def merge_intervals(intervals):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = [intervals[0]]
        for current in intervals[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        return merged

    for m in assigned_intervals:
        assigned_intervals[m] = merge_intervals(assigned_intervals[m])

    os.makedirs(output_dir, exist_ok=True)

    with open(input_filename, "r") as infile:
        header_line_1 = infile.readline()
        header_line_2 = infile.readline()
        lines = infile.readlines()

    for m, intervals in assigned_intervals.items():
        unique_lines = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            try:
                yr = int(parts[0])
                mon = int(parts[1])
                day = int(parts[2])
                date = datetime.date(yr, mon, day)
                if yr != year:
                    continue
                if any(start <= date <= end for start, end in intervals):
                    unique_lines.add(line.strip())
            except Exception:
                continue

        output_filename = os.path.join(
            output_dir,
            os.path.basename(input_filename).replace(".aut", f"_{m:02d}.aut")
        )
        with open(output_filename, "w") as outfile:
            outfile.write(header_line_1)
            outfile.write(header_line_2)
            for line in sorted(unique_lines):
                outfile.write(line + "\n")

        ranges_str = ", ".join(f"{start} to {end}" for start, end in intervals)
        print(f"Written {len(unique_lines)} lines for year {year}, month {m:02d} covering intervals: {ranges_str} -> {output_filename}")


out = r"C:\Users\fionu\OneDrive\Dokumente\Daten Fiona\AIUB\aut_files"
inp = r"C:\Users\fionu\OneDrive\Dokumente\Daten Fiona\AIUB\aut_files"

split_aut_file_distribute_missing(f"{inp}\\GEO_2019.aut", 2019, [2,5,11], out)
split_aut_file_distribute_missing(f"{inp}\\GTO_2019.aut", 2019, [2,5,11], out)
split_aut_file_distribute_missing(f"{inp}\\followup_2019.aut", 2019, [2,5,11], out)

split_aut_file_distribute_missing(f"{inp}\\GEO_2020.aut", 2020, [2,5,8,11], out)
split_aut_file_distribute_missing(f"{inp}\\GTO_2020.aut", 2020, [2,5,8,11], out)
split_aut_file_distribute_missing(f"{inp}\\followup_2020.aut", 2020, [2,5,8,11], out)

split_aut_file_distribute_missing(f"{inp}\\GEO_2021.aut", 2021, [2,5,8], out)
split_aut_file_distribute_missing(f"{inp}\\GTO_2021.aut", 2021, [2,5,8], out)
split_aut_file_distribute_missing(f"{inp}\\followup_2021.aut", 2021, [2,5,8], out)

split_aut_file_distribute_missing(f"{inp}\\GEO_2022.aut", 2022, [5,8,11], out)
split_aut_file_distribute_missing(f"{inp}\\GTO_2022.aut", 2022, [5,8,11], out)
split_aut_file_distribute_missing(f"{inp}\\followup_2022.aut", 2022, [5,8,11], out)

split_aut_file_distribute_missing(f"{inp}\\GEO_2023.aut", 2023, [2,5,8,11], out)
split_aut_file_distribute_missing(f"{inp}\\GTO_2023.aut", 2023, [2,5,8,11], out)
split_aut_file_distribute_missing(f"{inp}\\followup_2023.aut", 2023, [2,5,8,11], out)
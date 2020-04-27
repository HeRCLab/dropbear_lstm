# RTML data format converter script.
import sys
import argparse
import json
import csv


def parse_args():
    parser = argparse.ArgumentParser(description='RTML Data Format Converter Script')
    parser.add_argument('from_format', help="Data format to get data from. (labview|json|csv)")
    parser.add_argument('to_format', help="Data format to convert data to. (json|csv)")
    parser.add_argument('-f', type=str, help="Filename to read from.")
    parser.add_argument('--skip-lines', type=int, default=23, help="Lines of header text before the data starts. (default: 23)")

    return parser.parse_args()


class LabViewReader(object):
    def __init__(self, stream, header_lines=23):
        self.stream = stream
        self.date_time = ""
        # Skip lines leading up to the fields header.
        # As we iterate, pick up interesting metadata.
        for i in range(0, header_lines-1):
            line = self.stream.readline()
            if line.startswith("Date"):
                parts = line.split()
                self.date_time = parts[-1]
            elif line.startswith("Time"):
                parts = line.split()
                self.date_time += " " + parts[-1]
            elif line.startswith("Samples"):
                parts = line.split()
                self.sample_rate = int(parts[-1])
        # Read the header line.
        header = self.stream.readline()
        self.headers = header.split()
        if "Acceleration" not in self.headers:
            print("'Acceleration' header not found on last row.")

    def __iter__(self):
        return self

    def __next__(self):
        line = self.stream.readline()
        if line == '':
            raise StopIteration()
        readings = [float(x) for x in line.split()]
        return {k: v for (k, v) in zip(self.headers, readings)}


if __name__ == '__main__':
    dataset = {
        "acceleration_data": [],
        "accelerometer_sample_rate": None,
        "date_time": None,
        "time_acceleration_data": [],
        "time_accelerometer_aquire_start": None,
    }

    args = parse_args()

    f = sys.stdin
    if args.f is not None:
        f = open(args.f, 'r')

    # Read in data.
    if args.from_format == "labview":
        reader = LabViewReader(f, args.skip_lines)
        dataset["date_time"] = reader.date_time
        dataset["accelerometer_sample_rate"] = reader.sample_rate
        for row in reader:
            dataset["acceleration_data"].append(row["Acceleration"])
        # TODO: Pick up time axis, and force data.
    elif args.from_format == "json":
        dataset = json.loads(f.read())
    elif args.from_format == "csv":
        reader = csv.DictReader(f)
        times = []
        for row in reader:
            dataset["acceleration_data"].append(float(row["Observation"]))
            if len(times) < 2:
                times.append(float(row["Time"]))
        if len(times) == 2:
            dataset["accelerometer_sample_rate"] = 1.0 / (times[1])

    # Write out the converted data.
    if args.to_format == "json":
        print(json.dumps(dataset))
    elif args.to_format == "csv":
        print("Not implemented yet.")
        exit(1)
    f.close()

import json

INPUT_FILE = 'spice_input.json'
OUTPUT_PREFIX = 'spice_input_file'
NUM_SPLITS = 40

def main():
    with open(INPUT_FILE, 'r') as fp:
        data = json.load(fp)
    split_size = len(data) // NUM_SPLITS

    print "Data Size = %d\nSplit Size=%d" % (len(data), split_size)

    for i in range(NUM_SPLITS):
        values = data[i * split_size: (i+1) * split_size]
        with open('%s_%d.json' % (OUTPUT_PREFIX, i), 'w') as fp:
            json.dump(values, fp)

if __name__ == '__main__':
    main()

import json

INPUT_PREFIX = 'spice_output_file'
OUTPUT_FILE = 'spice_output_train.json'

START_IDX = 0
END_IDX = 40

def main():
    data = []
    for i in range(START_IDX, END_IDX):
        with open('%s_%d.json' % (INPUT_PREFIX, i), 'r') as fp:
            data += json.load(fp)

        print "Output Data Size = %d" % len(data)
    with open(OUTPUT_FILE, 'w') as fp:
        json.dump(data, fp)

if __name__ == '__main__':
    main()

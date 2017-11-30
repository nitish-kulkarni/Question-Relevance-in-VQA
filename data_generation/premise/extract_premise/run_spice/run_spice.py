import os

INPUT_PREFIX = 'spice_input_file'
OUTPUT_PREFIX = 'spice_output_file'

START_IDX = 5
END_IDX = 40

def main():
    for i in range(START_IDX, END_IDX):
        input_file = '%s_%d.json' % (INPUT_PREFIX, i)
        output_file = '%s_%d.json' % (OUTPUT_PREFIX, i)
        cmd = 'time java -Xmx8G -jar ../../SPICE-1.0/spice-*.jar %s -out %s -detailed' % (input_file, output_file)

        print 'Running cmd:\n%s' % cmd
        os.system(cmd)

if __name__ == '__main__':
    main()

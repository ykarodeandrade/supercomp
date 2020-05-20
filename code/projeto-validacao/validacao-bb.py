import sys
from grading_utils import IOTest

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: ./validacao-bb executavel')
        sys.exit(-1)

    t = IOTest(sys.argv[1], 'in_bb')
    t.main()

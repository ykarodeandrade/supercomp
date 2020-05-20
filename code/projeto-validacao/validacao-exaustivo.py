from grading_utils import IOTest
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: ./validacao-exaustivo executavel')
        sys.exit(-1)

    t = IOTest(sys.argv[1], 'in_exaustivo')
    t.main()

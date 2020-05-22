from collections import defaultdict, namedtuple
import argparse
import filecmp
import re
import os
import numpy as np
import subprocess
from dataclasses import dataclass
from typing import Optional

def get_file_contents(fname):
    with open(fname) as f:
        return f.read()

@dataclass
class TestConfiguration:
    input: str
    output: str
    stderr: str
    check_stderr: bool = True
    time_limit: Optional[int] = None

    @staticmethod
    def from_pattern(dir, pattern, check_stderr=True):
        tests = {}

        for entry in os.listdir(dir):
            if entry.startswith(pattern):
                problema = '_'.join(entry.split('_')[1:])
                entry = f'{dir}/{entry}'
                output_file = f'{dir}/out_{problema}'
                err_file = f'{dir}/err_{problema}'



                tests[entry] = TestConfiguration(
                                    get_file_contents(entry), 
                                    get_file_contents(output_file), 
                                    get_file_contents(err_file), 
                                    check_stderr=check_stderr)
        return tests

class ProgramTest:
    def __init__(self, cmd, tests={}):
        self.program_cmd = cmd
        self.tests = tests

    def main(self):
        pass_all = True
        for arq, test in self.tests.items():
            print(f'====================\nEntrada: {arq}')
            try:
                stdout, stderr = run_program(self.program_cmd, test.input, test.time_limit)
            except subprocess.TimeoutExpired:
                self.timeout_event(test)
            else:
                if not self.test_program_result(test, stdout, stderr):
                    pass_all = False
        print('====================\nValidated:', pass_all)
        
    def timeout_event(self, test):
        print(f'Timeout exceeded: {test.time_limit}')

    def test_program_result(self, test, 
                            stdout, stderr):
        return True

class IOTest(ProgramTest):
    def test_program_result(self, test, stdout, stderr):
        valido = valid_solution(test.input, stdout)
        print('Solução válida', valido)
        saida_ok = compare_outputs(test.output, stdout)
        print('Saída: ', saida_ok)
        err_ok = True
        if test.check_stderr:
            err_ok = compare_outputs(test.stderr, stderr)
            print('Verificações: ', err_ok)
        return valido and saida_ok and err_ok

class PerformanceTest(ProgramTest):
    def test_program_result(self, stdin, expected_stdout, expected_stderr, 
                            stdout, stderr, timeout):
        saida_ok = compare_outputs(stdout, expected_stdout)
        print('Saída:', saida_ok)
        return saida_ok


class bcolors:
    OKGREEN = '\033[92m' + u'\u2713' + '\033[0m'
    FAILRED = '\033[91m' + "x" + '\033[0m'
    INVYELL = '\033[33m' + "?" + '\033[0m'


def satisfaction(attr, prefs):
    sat = 0
    for i in range(len(attr)):
        sat += prefs[i, attr[i]]
    return sat


def parse_output(output):
    if not isinstance(output, list):
        output = output.split('\n')
    sat, opt = [int(x) for x in output[0].split()]
    attr = [int(x) for x in output[1].split()]

    return sat, opt, attr


def parse_input(input_txt):
    if not isinstance(input_txt, list):
        input_txt = input_txt.split('\n')
    n_alunos, n_projetos, n_choices = [int(p) for p in input_txt[0].split()]
    prefs = np.zeros((n_alunos, n_projetos), np.uint32)
    for i in range(n_alunos):
        projs = [int(c) for c in input_txt[i+1].split(' ')]
        for j, p in enumerate(projs):
            prefs[i, p] = pow(n_choices - j, 2)

    return prefs, n_choices

def list_all_input_files(preffix):
    inputs = []
    for entr in os.listdir('entradas'):
        if not entr.startswith(preffix):
            continue
        problema = '_'.join(entr.split('_')[1:])

        with open(f'entradas/out_{problema}') as f:
            saida_original = f.read().strip()

        with open(f'entradas/err_{problema}') as f:
            verificacoes_original = f.read().strip()

        with open(f'entradas/{entr}') as f:
            texto_entrada = f.read()

        inputs.append((entr, texto_entrada, saida_original, verificacoes_original))

    return inputs


def run_program(command, input_txt, time_limit=None):
    proc = subprocess.run([command], input=input_txt.encode('ascii'),
                          capture_output=True, env=os.environ, timeout=time_limit)
    out_proc = str(proc.stdout, 'ascii').strip()
    err_proc = str(proc.stderr, 'ascii').strip()
    return out_proc, err_proc


def compare_outputs(out1, out2, ignore_whitespace=True):
    if ignore_whitespace:
        out1_chunks = out1.split()
        out2_chunks = out2.split()
        return out1_chunks == out2_chunks
    else:
        return out1.strip() == out2.strip()

def check_format(data_out):
    """
    Função que verifica se o arquivo de saída esta conforme o formato especificado

    satisfacao opt
    pa1 pa2 pa3 ... pa(n_alunos)
    """
    return re.fullmatch(r"\d* (0|1)", data_out[0]) and re.fullmatch(
        r"(\d+ )+[\d]+", data_out[1]) and len(data_out) >= 2


def valid_solution(data_inp, data_out):
    """
    Função que verifica se a solução é válida (todo aluno tem projeto, 
    cada projeto tem exatamente 3 alunos).
    """
    data_out = data_out.split('\n')
    data_inp = data_inp.split('\n')
    st, opt = [int(x) for x in data_out[0].split(" ")]
    n_alun, n_proj, n_choices = [int(x) for x in data_inp[0].split(" ")]

    proj = defaultdict(lambda: [])
    alun = defaultdict(lambda: [])

    n_choices = int(data_inp[0].split()[2])
    for i, c in enumerate(data_out[1].split(" ")):
        alun[i] += [c]
        proj[c] += [i]

        try:
            st -= pow(n_choices - data_inp[i + 1].split(" ").index(c), 2)
        except:
            pass
        
    return not st and len(
        [len(x)
         for x in proj.values() if len(set(x)) == 3]) == n_proj and len(
             [len(x) for x in alun.values() if len(x) == 1]) == n_alun


def check_result(args):
    """
    Função que verifica se os valores e o formato do aquivo de saída do 
    aluno é igual ao arquivo de saida esperado. 
    """
    return filecmp.cmp(args.output, args.real_output, shallow=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Arquivo de entrada")
    parser.add_argument("output", help="Arquivo de saída")
    parser.add_argument("real_output",
                        help="Arquivo de saída correta",
                        nargs='?',
                        default=None)

    args = parser.parse_args()

    with open(args.input, "r") as f:
        data_inp = f.read().split("\n")

    with open(args.output, "r") as f:
        data_out = f.read().split("\n")

    if (args.real_output):
        tmp = bcolors.OKGREEN if check_result(args) else bcolors.FAILRED
        print(f"Result Output {tmp}")

    tmp = bcolors.OKGREEN if check_format(data_out) else bcolors.FAILRED
    print(f"Format Output {tmp}")

    if (tmp == bcolors.OKGREEN):
        tmp = bcolors.OKGREEN if check_solution(data_inp,
                                                data_out) else bcolors.FAILRED
        print(f"Valid Solution {tmp}")

    else:
        print(f"Valid Solution {bcolors.INVYELL}")


if __name__ == "__main__":
    main()

from grading_utils import ProgramTest, TestConfiguration, CheckStderrMixin, CheckOutputMixin
import numpy as np

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


def satisfaction(attr, prefs):
    sat = 0
    for i in range(len(attr)):
        sat += prefs[i, attr[i]]
    return sat


class SolucaoValidaMixin:
    def test_solucao_valida(self, test, stdout, stderr):
        sat, opt, attr = parse_output(stdout)
        prefs, n_choices = parse_input(test.input)
        return sat == satisfaction(attr, prefs)


class SatisfacaoOtimaMixin:
    def test_satisfacao_otima(self, test, stdout, stderr):
        sat, opt, attr = parse_output(stdout)
        esat, eopt, eattr = parse_output(test.output)
        prefs, n_choices = parse_input(test.input)
        return esat == sat

class ChecaSatisfacoesStderrMixin:
    def test_checa_satisfacoes_em_stderr(self, test, stdout, stderr):
        for i, l in enumerate(stderr.split('\n')):
            if l.startswith('Melhor:') or l.startswith('Inicial') or \
               l.startswith('Iter:'):

                sat, *attr = l.split()[1:]
                sat = int(sat)
                attr = [int(i) for i in attr]
                sat_comp = 0
                for k, p in enumerate(attr):
                    sat_comp += prefs[k, p]
                if sat != sat_comp:
                    print(f'Erro na verificação da linha {i}:')
                    print(f'Satisfação lida: {sat}')
                    print(f'Satisfação real: {sat_comp}')
                    return False
        
        return True


class TestePFEExaustivo(ProgramTest, SolucaoValidaMixin, CheckStderrMixin, 
                        CheckOutputMixin, ChecaSatisfacoesStderrMixin):
    pass
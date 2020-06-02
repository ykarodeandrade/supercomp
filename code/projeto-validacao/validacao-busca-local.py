import sys
import os
import subprocess
import re
from grading_utils import ProgramTest, valid_solution, parse_input, satisfaction
from grading_utils import TestConfiguration, parse_output, RepeaterTest


class BuscaLocalTest(ProgramTest):
    def test_solucao_valida(self, test, stdout, stderr):
        return valid_solution(test.input, stdout)

    def test_executa_ITER_vezes(self, test, stdout, stderr):
        m = re.findall('Inicial', stderr)
        return len(m) == int(test.environ['ITER'])

    def test_solucao_sempre_melhora(self, test, stdout, stderr):
        sempre_melhora = True
        sat_atual = -1
        for l in stderr.split('\n'):
            _, sat_next, *attr_next = l.split(' ')
            sat_next = int(sat_next)
            
            if sat_next < sat_atual:
                sempre_melhora = False

            sat_atual = sat_next
        
        return sempre_melhora

    def test_solucao_otimo_local(self, test, stdout, stderr):
        sat_atual, opt, attr_atual = parse_output(stdout)
        prefs, n_choices = parse_input(test.input)
    
        for i in range(prefs.shape[0]):
            for j in range(prefs.shape[0]):
                # NOTE: Isto está feito ruim de propósito para não entregar o algoritmo
                attr_teste = attr_atual.copy()
                attr_teste[i], attr_teste[j] = attr_teste[j], attr_teste[i]
                sat_teste = satisfaction(attr_teste, prefs)
                if sat_teste > sat_atual:
                    print('Troca entre', i, 'e', j, 'melhoraria solucao:', sat_teste)
                    return False
        
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: ./validacao-busca-local executavel')
        sys.exit(-1)

    tests = TestConfiguration.from_pattern('entradas', 'in_local', check_stderr=False,
                                           environ={'ITER': '1', 'SEED': '10'})
    t = BuscaLocalTest(sys.argv[1], tests)
    t.main()
    
    teste_grande = TestConfiguration.from_file('entradas/in_local_72_24_5', 'entradas/out_local_72_24_5', check_stderr=False,
                                           environ={'ITER': '10', 'SEED': '1'})
    r = RepeaterTest(sys.argv[1], teste_grande, 10)
    r.main()
    



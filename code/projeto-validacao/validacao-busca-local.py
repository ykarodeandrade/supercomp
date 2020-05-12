import sys
import os
import subprocess
import re
from grading_utils import list_all_input_files, valid_solution, parse_input, satisfaction, run_program

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: ./validacao-busca-local executavel')
        sys.exit(-1)
    
    os.environ['ITER'] = '1'
    os.environ['SEED'] = '10'

    tudo_ok = True

    nome_executavel = sys.argv[1]
    for entr in list_all_input_files('in_local_'):
        arq, inp, out, ver = entr

        print(f'====================\nEntrada: {arq}')
        out_proc, err_proc_all = run_program(nome_executavel, inp)
        print('Solução válida:', valid_solution(inp, out_proc))

        solucao_sempre_melhora = True
        err_proc = err_proc_all.split('\n')
        _, sat_atual, *attr_atual = err_proc[0].split(' ')
        sat_atual = int(sat_atual)
        attr_atual = [int(x) for x in ' '.join(attr_atual).split()]
        for l in err_proc[1:]:
            _, sat_next, *attr_next = l.split(' ')
            sat_next = int(sat_next)
            attr_next = [int(x) for x in ' '.join(attr_next).split()]
            if sat_next < sat_atual:
                solucao_sempre_melhora = False

            sat_atual = sat_next
            attr_atual = attr_next

        print('Solução melhora a cada iteração', solucao_sempre_melhora)
        prefs, n_choices = parse_input(inp)
        n_alunos = prefs.shape[0]
        pode_melhorar = False
        for i in range(n_alunos):
            for j in range(n_alunos):
                # NOTE: Isto está feito ruim de propósito para não entregar o algoritmo
                attr_teste = attr_atual.copy()
                attr_teste[i], attr_teste[j] = attr_teste[j], attr_teste[i]
                sat_teste = satisfaction(attr_teste, prefs)
                if sat_teste > sat_atual:
                    pode_melhorar = True
                    print('Troca entre', i, 'e', j, 'melhoraria solucao:', sat_teste)
        
        print('Solução é ótimo local', not pode_melhorar)
        m = re.findall('Inicial', err_proc_all)
        print('Só é feita uma iteração', len(m) == 1)
        out_proc2, err_proc_all2 = run_program(nome_executavel, inp)
        print('Uma segunda execução retorna os mesmos resultados', out_proc == out_proc2 and
               err_proc_all == err_proc_all2)


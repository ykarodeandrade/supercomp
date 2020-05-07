import sys
import os
import subprocess
from validator import list_all_input_files, valid_solution, parse_input, satisfaction

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
        proc = subprocess.run([nome_executavel], input=inp.encode('ascii'),
                              capture_output=True, env=os.environ)
        out_proc = str(proc.stdout, 'ascii').strip()
        err_proc = str(proc.stderr, 'ascii').strip().split('\n')
        print('Solução válida:', valid_solution(inp, out_proc))

        solucao_sempre_melhora = True
        _, sat_atual, *attr_atual = err_proc[0].split(' ')
        sat_atual = int(sat_atual)
        attr_atual = [int(x) for x in ' '.join(attr_atual)[1:-1].split()]
        for l in err_proc[1:]:
            _, sat_next, *attr_next = l.split(' ')
            sat_next = int(sat_next)
            attr_next = [int(x) for x in ' '.join(attr_next)[1:-1].split()]
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
        
        print('Solução é ótimo local', not pode_melhorar)
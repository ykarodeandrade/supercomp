import subprocess
import sys

from grading_utils import valid_solution, list_all_input_files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Uso: ./validacao-exaustivo executavel')
        sys.exit(-1)

    nome_executavel = sys.argv[1]

    for entr in list_all_input_files('in_exaustivo_'):
        arq, texto_entrada, saida_original, verificacoes_original = entr

        print(f'====================\nEntrada: {arq}')
        proc = subprocess.run([nome_executavel],
                              input=texto_entrada.encode('ascii'),
                              capture_output=True)
        texto_saida = str(proc.stdout, 'ascii').strip()
        texto_verificacoes = str(proc.stderr, 'ascii').strip()

        if valid_solution(texto_entrada, texto_saida):
            print('Solução válida')

        print('Saída: ', texto_saida == saida_original)
        print('Verificações: ', texto_verificacoes == verificacoes_original)

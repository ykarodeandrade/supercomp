

# Burocracias 

## Turma B

* Horários: 
    * TER - 09:45
    * SEX - 13:30
    * Atendimento SEX - 15:30

??? note "Alunos"
    **TBA**    


??? note "Plano de aulas"



---------------------------------------------------------------------------ValueError                                Traceback (most recent call last)<ipython-input-1-a0a686815bf9> in <module>
      1 t1 = pd.read_excel('plano-de-aulas-t1.xlsx')
----> 2 t1['Data'] = t1['Data'].apply(lambda x: x.strftime('%d/%m'))
      3 print('\n'.join(['    %s'%l for l in tabulate.tabulate(t1[['Data', 'Questão/Problema',
      4 'Conteúdo/Atividade']], headers=['Data', 'Questão/Problema',
      5 'Conteúdo/Atividade'], tablefmt='pipe', showindex=False).split('\n')  ]))
~/base/lib/python3.8/site-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, **kwds)
   4106             else:
   4107                 values = self.astype(object)._values
-> 4108                 mapped = lib.map_infer(values, f, convert=convert_dtype)
   4109 
   4110         if len(mapped) and isinstance(mapped[0], Series):
pandas/_libs/lib.pyx in pandas._libs.lib.map_infer()
<ipython-input-1-a0a686815bf9> in <lambda>(x)
      1 t1 = pd.read_excel('plano-de-aulas-t1.xlsx')
----> 2 t1['Data'] = t1['Data'].apply(lambda x: x.strftime('%d/%m'))
      3 print('\n'.join(['    %s'%l for l in tabulate.tabulate(t1[['Data', 'Questão/Problema',
      4 'Conteúdo/Atividade']], headers=['Data', 'Questão/Problema',
      5 'Conteúdo/Atividade'], tablefmt='pipe', showindex=False).split('\n')  ]))
pandas/_libs/tslibs/nattype.pyx in pandas._libs.tslibs.nattype._make_error_func.f()
ValueError: NaTType does not support strftime


## Turma A

* Horários: 
    * QUA - 13:30
    * SEX - 07:30
    * Atendimento SEX - 09:30

??? note "Alunos"
    **TBA**



??? details "Plano de aulas"




---------------------------------------------------------------------------ValueError                                Traceback (most recent call last)<ipython-input-1-b03f8bd81b15> in <module>
      1 t2 = pd.read_excel('plano-de-aulas-t2.xlsx')
----> 2 t2['Data'] = t2['Data'].apply(lambda x: x.strftime('%d/%m'))
      3 print('\n'.join(['    %s'%l for l in tabulate.tabulate(t2[['Data', 'Questão/Problema',
      4 'Conteúdo/Atividade']], headers=['Data', 'Questão/Problema',
      5 'Conteúdo/Atividade'], tablefmt='pipe', showindex=False).split('\n')]))
~/base/lib/python3.8/site-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, **kwds)
   4106             else:
   4107                 values = self.astype(object)._values
-> 4108                 mapped = lib.map_infer(values, f, convert=convert_dtype)
   4109 
   4110         if len(mapped) and isinstance(mapped[0], Series):
pandas/_libs/lib.pyx in pandas._libs.lib.map_infer()
<ipython-input-1-b03f8bd81b15> in <lambda>(x)
      1 t2 = pd.read_excel('plano-de-aulas-t2.xlsx')
----> 2 t2['Data'] = t2['Data'].apply(lambda x: x.strftime('%d/%m'))
      3 print('\n'.join(['    %s'%l for l in tabulate.tabulate(t2[['Data', 'Questão/Problema',
      4 'Conteúdo/Atividade']], headers=['Data', 'Questão/Problema',
      5 'Conteúdo/Atividade'], tablefmt='pipe', showindex=False).split('\n')]))
pandas/_libs/tslibs/nattype.pyx in pandas._libs.tslibs.nattype._make_error_func.f()
ValueError: NaTType does not support strftime


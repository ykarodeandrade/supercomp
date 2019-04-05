// referencia https://ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/
#include <stdlib.h>

// Estrutura de dados para as imagens PGM
typedef struct _PGMData {
   int row;
   int col;
   int max_gray;
   int *matrix;
} PGMData;

// Rotina para ignorar comentarios em arquivos PGM
void SkipComments(FILE *fp);

// Ajusta informacoes e aloca dados para estrtura de armazenamento de imagens PGM
void createImage(PGMData *data, int row, int col, int max_gray, int channels);

// Executa a leitura de um arquivo PGM
void readImagePGM(const char *file_name, PGMData *data);
void readImagePPM(const char *file_name, PGMData *data, int channels);

// Grava um arquivo PGM
void writeImagePGM(const char *filename, const PGMData *data, unsigned char binary);
void writeImagePPM(const char *filename, const PGMData *data, int channels, unsigned char binary);



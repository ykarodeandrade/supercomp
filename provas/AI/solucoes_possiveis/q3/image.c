// https://ugurkoltuk.wordpress.com/2010/03/04/an-extreme-simple-pgm-io-api/
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "image.h" 

// Rotina para ignorar comentarios em arquivos PGM
void SkipComments(FILE *fp)
{
   int ch;
   char line[256];
   while ((ch = fgetc(fp)) != EOF && isspace(ch)) { }
   if (ch == '#') {
      fgets(line, sizeof(line), fp);
      SkipComments(fp);
   } else fseek(fp, -1, SEEK_CUR);
}

// Ajusta informacoes e aloca dados para estrtura de armazenamento de imagens PGM
void createImage(PGMData *data, int row, int col, int max_gray, int channels)
{
   data->row = row;
   data->col = col;
   data->max_gray = max_gray;
   data->matrix = (int *)malloc(sizeof(int) * data->row * data->col * channels);
}

// Executa a leitura de um arquivo PGM
void readImagePGM(const char *file_name, PGMData *data) {

   FILE *pgmFile;
   char version[3];
   int i, j;
   int tmp;
   unsigned char binary = 0;
   pgmFile = fopen(file_name, "rb");
   if (pgmFile == NULL) {
      perror("Falha ao abrir arquivo para leitura");
      exit(EXIT_FAILURE);
   }
   fgets(version, sizeof(version), pgmFile);
   if (!strcmp(version, "P5")) binary = 1;
   else if (!strcmp(version, "P2")) binary = 0;
   else {
      fprintf(stderr, "Erro ao identificar arquivo PGM!\n");
      exit(EXIT_FAILURE);
   }
   SkipComments(pgmFile);
   fscanf(pgmFile, "%d", &data->col);
   fscanf(pgmFile, "%d", &data->row);
   fscanf(pgmFile, "%d", &data->max_gray);
   fgetc(pgmFile);
 
   data->matrix = (int *)malloc(sizeof(int) * data->row * data->col);
   for (i = 0; i < data->row; ++i)
      for (j = 0; j < data->col; ++j) {
         if(binary) {
            data->matrix[i*data->col+j] = fgetc(pgmFile);
         } else {
            fscanf(pgmFile, "%d", &tmp);
            data->matrix[i*data->col+j] = tmp;
         }
      }
 
   fclose(pgmFile);
}

// Executa a leitura de um arquivo PGM
void readImagePPM(const char *file_name, PGMData *data, int channels) {
   
   FILE *ppmFile;
   char version[3];
   int i, j, k;
   //char tmp[8];
   int tmp;
   unsigned char ascii = 0;
   ppmFile = fopen(file_name, "rb");
   if (ppmFile == NULL) {
      perror("Falha ao abrir arquivo para leitura");
      exit(EXIT_FAILURE);
   }
   fgets(version, sizeof(version), ppmFile);
   if (!strcmp(version, "P6")) ascii = 1;
   else if (!strcmp(version, "P3")) ascii = 0;
   else {
      fprintf(stderr, "Erro ao identificar arquivo PPM!\n");
      exit(EXIT_FAILURE);
   }
   SkipComments(ppmFile);
   fscanf(ppmFile, "%d", &data->col);
   fscanf(ppmFile, "%d", &data->row);
   fscanf(ppmFile, "%d", &data->max_gray);
   fgetc(ppmFile);
 
   data->matrix = (int *)malloc(sizeof(int) * data->row * data->col * channels);
   for (i = 0; i < data->row; ++i)
      for (j = 0; j < data->col; ++j) {
         for (k = 0; k < channels; ++k) {
            if(ascii) {
               data->matrix[(i*data->col*channels)+(j*channels)+k] = fgetc(ppmFile);
            } else {
               fscanf(ppmFile, "%d", &tmp);
               data->matrix[(i*data->col*channels)+(j*channels)+k] = tmp;
            }
         }
      }
 
   fclose(ppmFile);
}

// Grava um arquivo PGM
void writeImagePGM(const char *filename, const PGMData *data, unsigned char binary) {

   FILE *pgmFile;
   int i, j;
 
   pgmFile = fopen(filename, "w");
   if (pgmFile == NULL) {
      perror("Falha ao abrir arquivo para escrita");
      exit(EXIT_FAILURE);
   }
 
   fprintf(pgmFile, (binary?"P5\n":"P2\n"));
   fprintf(pgmFile, "%d %d\n", data->col, data->row);
   fprintf(pgmFile, "%d\n", data->max_gray);
 
   for (i = 0; i < data->row; ++i) {
      for (j = 0; j < data->col; ++j) {
         if(binary) {
            fputc(data->matrix[i*data->col+j], pgmFile);   
         } else {
            fprintf(pgmFile, " %d ", data->matrix[i*data->col+j]);
         }
      }
      if(!binary) {
         fprintf(pgmFile, "\n");
      }
   }
         
   fclose(pgmFile);
   free(data->matrix);
}

// Grava um arquivo PPM
void writeImagePPM(const char *filename, const PGMData *data, int channels, unsigned char binary) {
   FILE *ppmFile;
   int i, j, k;
 
   ppmFile = fopen(filename, "w");
   if (ppmFile == NULL) {
      perror("Falha ao abrir arquivo para escrita");
      exit(EXIT_FAILURE);
   }
 
   fprintf(ppmFile, (binary?"P6\n":"P3\n"));
   fprintf(ppmFile, "%d %d\n", data->col, data->row);
   fprintf(ppmFile, "%d\n", data->max_gray);
 
   for (i = 0; i < data->row; ++i) {
      for (j = 0; j < data->col; ++j) {
         for (k = 0; k < channels; ++k) {
            if(binary) {
               fputc(data->matrix[(i*data->col*channels)+(j*channels)+k], ppmFile);
            } else {
               fprintf(ppmFile, "%d ", data->matrix[(i*data->col*channels)+(j*channels)+k]);
            } 
         }
         if(!binary) fprintf(ppmFile, "\n");

      }
      if(!binary) {
         fprintf(ppmFile, "\n");
      }
   }
         
   fclose(ppmFile);
   free(data->matrix);
}